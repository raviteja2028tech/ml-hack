"""
PathSnap Phase 3 - per-trip pipeline: load -> candidates -> features -> (labels).

``process_trip`` is used by ``scripts/build_phase3_dataset.py`` (all trips),
``scripts/benchmark_phase3.py`` and the unit tests. Labels are computed only if
``with_labels=True`` and the trip has a route, and always after the features,
which never receive the route.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.candidate_generator import ArcGeometry, CandidateConfig, generate_candidates
from src.feature_engineering import FEATURE_COLUMNS, FeatureConfig, build_arc_graph, compute_point_kinematics, build_candidate_features
from src.pathsnap_loader import load_trip

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "Student folder"


@dataclass(frozen=True)
class Phase3Config:
    candidate: CandidateConfig = field(default_factory=CandidateConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    label: "LabelConfig" = None  # filled lazily to keep the import of route_labeler out of feature paths

    def to_dict(self) -> Dict:
        from src.route_labeler import LabelConfig

        return {
            "candidate": asdict(self.candidate),
            "feature": asdict(self.feature),
            "label": asdict(self.label or LabelConfig()),
        }


@dataclass
class TripResult:
    trip_id: str
    partition: str
    candidates: pd.DataFrame  # one row per (point, candidate arc): ids + features (+ labels)
    points: pd.DataFrame  # one row per GPS point
    stats: Dict


def process_trip(
    trip_id: str,
    partition: Optional[str] = None,
    base_dir: Path = DEFAULT_BASE_DIR,
    config: Phase3Config = Phase3Config(),
    with_labels: bool = True,
    drop_route_before_features: bool = True,
    trip=None,
    label_sensitivity_weights=(),
) -> TripResult:
    timings = {}
    t0 = time.perf_counter()
    if trip is None:
        trip = load_trip(trip_id, base_dir=base_dir, partition=partition)
    timings["load_s"] = time.perf_counter() - t0

    # Keep the route away from every feature computation.
    route = trip.route
    if drop_route_before_features:
        trip.route = None

    t = time.perf_counter()
    geom = ArcGeometry.from_trip(trip)
    graph = build_arc_graph(geom)
    tr = trip.track_df
    px = tr["x_m"].to_numpy(np.float64)
    py = tr["y_m"].to_numpy(np.float64)
    ts = tr["timestamp"].to_numpy(np.float64)
    timings["graph_s"] = time.perf_counter() - t

    t = time.perf_counter()
    cands, cstats = generate_candidates(geom, px, py, config.candidate, return_stats=True)
    timings["candidates_s"] = time.perf_counter() - t

    t = time.perf_counter()
    kin = compute_point_kinematics(px, py, ts, config.feature)
    feats = build_candidate_features(cands, kin, geom, graph, px, py, config.feature)
    timings["features_s"] = time.perf_counter() - t

    pi = cands["point_idx"].to_numpy(np.int64)
    cand_df = pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "point_idx": cands["point_idx"].to_numpy(np.int32),
            "timestamp": ts[pi],
            "segment_id": kin["segment_id"].to_numpy(np.int32)[pi],
            "arc_id": cands["arc_id"].to_numpy(np.int32),
        }
    )
    cand_df = pd.concat([cand_df, feats.reset_index(drop=True)], axis=1)

    n_cand = np.bincount(pi, minlength=len(tr)).astype(np.int16)
    points_df = pd.DataFrame(
        {
            "trip_id": trip.trip_id,
            "point_idx": np.arange(len(tr), dtype=np.int32),
            "lon": tr["lon"].to_numpy(),
            "lat": tr["lat"].to_numpy(),
            "timestamp": ts,
            "segment_id": kin["segment_id"].to_numpy(np.int32),
            "n_candidates": n_cand,
            "speed_mps": kin["speed_mps"].to_numpy(np.float32),
            "heading_valid": kin["heading_valid"].to_numpy(np.int8),
        }
    )

    stats: Dict = {
        "trip_id": trip.trip_id,
        "partition": trip.partition,
        "n_points": int(len(tr)),
        "n_arcs": int(geom.n_arcs),
        "n_segments": int(kin["segment_id"].max() + 1) if len(kin) else 0,
        "candidate_stats": cstats,
    }

    if with_labels and route is not None:
        from src.route_labeler import LabelConfig, align_track_to_route, boundary_margin, label_candidates, label_quality

        lcfg = config.label or LabelConfig()
        t = time.perf_counter()
        hdg = kin["heading_deg"].to_numpy()
        hv = kin["heading_valid"].to_numpy()
        align = align_track_to_route(px, py, hdg, hv, geom, route, lcfg)
        margin = boundary_margin(px, py, hdg, hv, geom, route, align, lcfg)
        cand_lab, pt_lab = label_candidates(cands, align, margin, geom, lcfg)
        timings["labels_s"] = time.perf_counter() - t

        cand_df = pd.concat([cand_df, cand_lab.reset_index(drop=True)], axis=1)
        points_df = pd.concat([points_df, pt_lab], axis=1)
        q = label_quality(
            pt_lab, geom, route, px, py, ts, align["align_t"], kin["segment_id"].to_numpy(), config.candidate.radius_m
        )
        # Label-free heuristic baselines (what Phase 4 must beat), on points that have a positive:
        #   nearest  = smallest distance (ties on two-way roads broken by arc_id -> ~coin flip)
        #   anchor   = distance + heading penalty (same rule the hop features use)
        points_df["nearest_is_true"] = np.int8(0)
        points_df["anchor_is_true"] = np.int8(0)
        if len(cand_df):
            has_pos = points_df["has_positive"].to_numpy() == 1
            nearest = cand_df[cand_df["dist_rank"] == 0]
            points_df.loc[nearest["point_idx"].to_numpy(), "nearest_is_true"] = nearest["label"].to_numpy()
            score = cand_df["dist_m"].to_numpy(np.float64) + config.feature.anchor_heading_weight_m * (
                1.0 - cand_df["heading_cos"].to_numpy(np.float64)
            )
            order = np.lexsort((score, cand_df["point_idx"].to_numpy()))
            pts_sorted = cand_df["point_idx"].to_numpy()[order]
            first = order[np.r_[0, np.flatnonzero(np.diff(pts_sorted)) + 1]]
            points_df.loc[cand_df["point_idx"].to_numpy()[first], "anchor_is_true"] = cand_df["label"].to_numpy()[first]
            if has_pos.any():
                q["nearest_candidate_top1_on_labelled"] = float(points_df["nearest_is_true"].to_numpy()[has_pos].mean())
                q["anchor_heuristic_top1_on_labelled"] = float(points_df["anchor_is_true"].to_numpy()[has_pos].mean())

        # Label stability: re-align with a different heading weight; how many labels change?
        on = points_df["label_status"].astype(str).to_numpy() != "off_route"
        for w_alt in label_sensitivity_weights:
            alt = align_track_to_route(px, py, hdg, hv, geom, route, replace(lcfg, heading_weight_m=w_alt))
            same = alt["true_arc_id"] == align["true_arc_id"]
            q[f"label_agreement_w{int(w_alt)}"] = float(same[on].mean()) if on.any() else float("nan")
        q["positives"] = int(cand_df["label"].sum()) if len(cand_df) else 0
        q["label_equiv_rows"] = int(cand_df["label_equiv"].sum()) if len(cand_df) else 0
        q["points_with_equiv_twin"] = int(cand_df.loc[cand_df["label_equiv"] == 1, "point_idx"].nunique()) if len(cand_df) else 0
        q["max_positives_per_point"] = int(cand_df.groupby("point_idx")["label"].sum().max()) if len(cand_df) else 0
        q["near_tie_negative_points"] = _near_tie_points(cand_df)
        q["route_connectivity"] = float(
            np.mean(geom.to_node[np.asarray(route)[:-1]] == geom.from_node[np.asarray(route)[1:]])
        ) if len(route) > 1 else 1.0
        stats["label_quality"] = q

    timings["total_s"] = time.perf_counter() - t0
    stats["timings"] = timings
    stats["n_candidate_rows"] = int(len(cand_df))
    return TripResult(trip.trip_id, trip.partition, cand_df, points_df, stats)


def _near_tie_points(cand_df: pd.DataFrame, dist_tol_m: float = 0.5, cos_tol: float = 0.05) -> int:
    """
    Points whose positive candidate has a negative (non-twin) candidate that is
    practically indistinguishable at that instant: same distance within ``dist_tol_m``
    and same heading agreement within ``cos_tol``. Usually a GPS point sitting on a
    junction node. Such points need temporal context to be classified.
    """
    pos = cand_df.loc[cand_df["label"] == 1, ["point_idx", "dist_m", "heading_cos"]]
    if pos.empty:
        return 0
    neg = cand_df.loc[(cand_df["label"] == 0) & (cand_df["label_equiv"] == 0), ["point_idx", "dist_m", "heading_cos"]]
    j = neg.merge(pos, on="point_idx", suffixes=("", "_pos"))
    tie = (np.abs(j["dist_m"] - j["dist_m_pos"]) < dist_tol_m) & (np.abs(j["heading_cos"] - j["heading_cos_pos"]) < cos_tol)
    return int(j.loc[tie, "point_idx"].nunique())
