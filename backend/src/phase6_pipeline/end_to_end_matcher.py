"""
Phase 6: end-to-end map matching for one trip.

    GPS track + road network (one trip only, nothing else is reprocessed)
      -> candidate arcs (Phase 3, exact radius search)
      -> 42 features (Phase 3, computed with the reference route removed)
      -> saved LightGBM predict_proba (Phase 4 model, artifacts/models/lightgbm.joblib)
      -> ML-only argmax per point  and  HMM/Viterbi sequence decoding (Phase 5)
      -> route geometry (lat/lon) + metrics (reference route used only for evaluation/display)

CLI:
    python -m src.phase6_pipeline.end_to_end_matcher 00000052
    python -m src.phase6_pipeline.end_to_end_matcher 00000052 --out results/demo_00000052.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np

from src.phase1_eda.pathsnap_loader import load_trip
from src.phase3_features.candidate_generator import ArcGeometry
from src.phase3_features.feature_engineering import FEATURE_COLUMNS
from src.phase3_features.phase3_pipeline import process_trip
from src.phase5_hmm.evaluate_hmm import compress, continuity, point_accuracy, route_overlap
from src.phase5_hmm.hmm_decoder import HMMConfig, HMMRouteDecoder, ml_only_argmax
from src.phase5_hmm.transition_model import RoadGraph

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "Student folder"
DEFAULT_MODEL = ROOT / "artifacts" / "models" / "lightgbm.joblib"
PARTITIONS = ("train", "student_test")
TRIP_ID_RE = re.compile(r"^\d{8}$")


def list_trips(partition: str = "student_test", base_dir: Path = RAW) -> List[str]:
    if partition not in PARTITIONS:
        raise ValueError(f"unknown partition {partition!r}")
    d = base_dir / partition
    if not d.exists():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and TRIP_ID_RE.match(p.name) and (p / f"{p.name}.track").exists())


def _polylines(arc_seq: List[int], fr: np.ndarray, to: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> List[List[List[float]]]:
    """Merge a compressed arc sequence into connected [lat, lon] polylines."""
    lines: List[List[List[float]]] = []
    prev = None
    for a in arc_seq:
        u, v = int(fr[a]), int(to[a])
        if prev is not None and int(to[prev]) == u:
            lines[-1].append([round(float(lat[v]), 6), round(float(lon[v]), 6)])
        else:
            lines.append([[round(float(lat[u]), 6), round(float(lon[u]), 6)], [round(float(lat[v]), 6), round(float(lon[v]), 6)]])
        prev = a
    return lines


class PathSnapPipeline:
    def __init__(self, model_path: Optional[Path] = None, hmm_config: HMMConfig = HMMConfig(), base_dir: Path = RAW):
        self.model_path = Path(model_path) if model_path else DEFAULT_MODEL
        bundle = joblib.load(self.model_path)
        if isinstance(bundle, dict):
            self.model, features, self.model_name = bundle["model"], list(bundle["features"]), bundle.get("name", "model")
        else:  # bare estimator
            self.model, features, self.model_name = bundle, list(FEATURE_COLUMNS), self.model_path.stem
        if features != list(FEATURE_COLUMNS):
            raise ValueError("model feature list does not match src.phase3_features.FEATURE_COLUMNS (order/name mismatch)")
        self.features = features
        self.hmm_config = hmm_config
        self.base_dir = Path(base_dir)
        try:
            self.model_file = self.model_path.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            self.model_file = self.model_path.name

    def run(self, trip_id: str, partition: str = "student_test", evaluate: bool = True, include_geometry: bool = True) -> Dict:
        if not TRIP_ID_RE.match(str(trip_id)):
            raise ValueError(f"invalid trip id {trip_id!r}")
        if trip_id not in list_trips(partition, self.base_dir):
            raise FileNotFoundError(f"trip {trip_id} not found in {partition}")
        timing: Dict[str, float] = {}
        t0 = time.perf_counter()

        trip = load_trip(trip_id, base_dir=self.base_dir, partition=partition)
        ref_route = [int(a) for a in trip.route] if trip.route is not None else None
        timing["load"] = (time.perf_counter() - t0) * 1e3

        # Features are computed with trip.route set to None inside process_trip; labels (if requested)
        # are computed afterwards from the route and kept in separate columns used only for evaluation.
        do_eval = evaluate and ref_route is not None
        res = process_trip(trip_id, partition=partition, base_dir=self.base_dir, trip=trip, with_labels=do_eval)
        tt = res.stats["timings"]
        timing["candidates"] = tt["graph_s"] * 1e3 + tt["candidates_s"] * 1e3
        timing["features"] = tt["features_s"] * 1e3
        cands = res.candidates
        pts = res.points
        n_points = len(pts)

        t = time.perf_counter()
        X = cands[self.features].to_numpy(np.float32)
        prob = self.model.predict_proba(X)[:, 1] if len(X) else np.empty(0)
        timing["model"] = (time.perf_counter() - t) * 1e3
        inf = cands[["point_idx", "arc_id", "proj_t"]].copy()
        inf["prob"] = prob

        t = time.perf_counter()
        geom = ArcGeometry.from_trip(trip)
        graph = RoadGraph.from_geometry(geom)
        px = trip.track_df["x_m"].to_numpy(np.float64)
        py = trip.track_df["y_m"].to_numpy(np.float64)
        ts = trip.track_df["timestamp"].to_numpy(np.float64)
        seg = pts["segment_id"].to_numpy(np.int64)
        ml_arc = ml_only_argmax(inf, n_points)
        dec = HMMRouteDecoder(graph, self.hmm_config).decode(inf, px, py, ts, seg)
        hmm_arc = dec.arc_id
        timing["hmm"] = (time.perf_counter() - t) * 1e3
        timing["inference_total"] = (time.perf_counter() - t0) * 1e3 - (tt.get("labels_s", 0.0) * 1e3)

        # validation: one prediction per point that has candidates, always one of its own candidates
        has_c = np.zeros(n_points, bool)
        has_c[inf["point_idx"].to_numpy(np.int64)] = True
        for name, arr in (("ml_only", ml_arc), ("hmm", hmm_arc)):
            if not np.array_equal(arr >= 0, has_c):
                raise RuntimeError(f"{name}: prediction count does not match points with candidates")
        cand_keys = set(zip(inf["point_idx"].tolist(), inf["arc_id"].tolist()))
        p_ok = np.flatnonzero(has_c)
        if not all((int(p), int(hmm_arc[p])) in cand_keys for p in p_ok):
            raise RuntimeError("hmm predicted an arc outside the point's candidate set")

        fr, to = geom.from_node, geom.to_node
        out: Dict = {
            "trip_id": trip_id,
            "partition": partition,
            "model": self.model_name,
            "n_points": int(n_points),
            "n_points_with_candidates": int(has_c.sum()),
            "n_candidate_rows": int(len(inf)),
            "n_arcs_in_network": int(geom.n_arcs),
            "points_changed_by_hmm": int(((ml_arc != hmm_arc) & has_c).sum()),
            "hmm_diagnostics": {
                "chains": dec.n_chains,
                "no_candidate_points": dec.n_no_candidates,
                "disconnected_steps": dec.n_disconnected_steps,
                "dijkstra_runs": graph.n_dijkstra,
            },
            "timing_ms": {k: round(v, 1) for k, v in timing.items()},
        }
        ml_seq, hmm_seq = compress(ml_arc), compress(hmm_arc)
        out["ml_only_route_arcs"] = ml_seq
        out["hmm_route_arcs"] = hmm_seq

        if include_geometry:
            nodes = trip.nodes_df.sort_values("node_id")
            lat, lon = nodes["lat"].to_numpy(), nodes["lon"].to_numpy()
            out["gps"] = [[round(float(a), 6), round(float(b), 6)] for a, b in zip(trip.track_df["lat"], trip.track_df["lon"])]
            out["gps_timestamps"] = [float(x) for x in ts]
            out["ml_only_arc_per_point"] = ml_arc.tolist()
            out["hmm_arc_per_point"] = hmm_arc.tolist()
            out["ml_only_geometry"] = _polylines(ml_seq, fr, to, lat, lon)
            out["hmm_geometry"] = _polylines(hmm_seq, fr, to, lat, lon)
            out["reference_geometry"] = _polylines(compress(ref_route), fr, to, lat, lon) if ref_route else None

        if do_eval:
            labelled = pts["label_status"].astype(str).isin(["ok", "boundary"]).to_numpy()
            lab = cands[["point_idx", "arc_id", "label", "label_equiv"]]
            metrics = {}
            for name, arr, seq in (("ml_only", ml_arc, ml_seq), ("hmm", hmm_arc, hmm_seq)):
                m = point_accuracy(arr, lab, labelled)
                m.update(continuity(arr, seg, fr, to))
                m.update(route_overlap(seq, ref_route, geom.length_m))
                metrics[name] = m
            metrics["reference_route_arcs"] = len(ref_route)
            metrics["labelled_points"] = int(labelled.sum())
            out["metrics"] = metrics
            out["timing_ms"]["evaluation_labels"] = round(tt.get("labels_s", 0.0) * 1e3, 1)
        else:
            out["metrics"] = None
        return out


def main(argv=None) -> Dict:
    ap = argparse.ArgumentParser(description="Run PathSnap map matching on one trip")
    ap.add_argument("trip_id")
    ap.add_argument("--partition", default="student_test", choices=PARTITIONS)
    ap.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the full JSON result here")
    args = ap.parse_args(argv)
    pipe = PathSnapPipeline(args.model)
    r = pipe.run(args.trip_id, args.partition, evaluate=not args.no_eval, include_geometry=args.out is not None)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(r), encoding="utf-8")
    summary = {k: r[k] for k in ("trip_id", "n_points", "n_points_with_candidates", "n_candidate_rows",
                                  "points_changed_by_hmm", "hmm_diagnostics", "timing_ms", "metrics")}
    print(json.dumps(summary, indent=2))
    return r


if __name__ == "__main__":
    main()
