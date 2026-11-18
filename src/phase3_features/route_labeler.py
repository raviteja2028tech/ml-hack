"""
PathSnap Phase 3 - Supervised labels from the reference ``.route``.

The ``.route`` file is an ordered list of traversed arc_ids. It is not one row
per GPS point:
    * a slow or stopped vehicle emits many points on one arc;
    * short arcs can be crossed between two 1 Hz samples and get no point;
    * routes can revisit an arc (loops) and contain U-turns on two-way roads;
    * the GPS track can extend past either end of the route (trip 54 keeps
      recording for ~4.6 h after the route ends, ~100-200 m off-network).

Method - monotone alignment (subsequence DTW) of GPS points to route positions:
    Choose a non-decreasing route position k_i for each point i that minimises
        sum_i  min(d(p_i, arc[route[k_i]]), dist_cap)
             + w_h * (1 - cos(heading_i - bearing(route[k_i]))) / 2 * heading_valid_i
    Monotonicity is what separates a revisit or the outbound and return legs of
    a U-turn. The heading term breaks the tie between geometrically identical
    opposite-direction arcs, and is used only when the heading was measured
    (vehicle moving). The path may start and end at any route position. This
    is O(N*K) time via prefix minima, and O(N*K) small ints for backpointers.

Per-point label status:
    ok        - aligned arc is a spatial candidate, unambiguous
    boundary  - an adjacent route arc (+-2 positions) costs within
                ``boundary_margin_m`` of the assigned one. The point sits at a
                junction, so either label is defensible.
    off_route - aligned arc is NOT in the point's candidate set (candidate
                miss): GPS outlier, off-network parking, or track outside the
                route span.

Leakage note: this module is the only place that reads the route. Feature code
never imports it. Labels are attached as separate ``label*`` columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.candidate_generator import ArcGeometry, angle_diff_deg, point_segment_distance

LABEL_STATUSES = ["ok", "boundary", "off_route"]


@dataclass(frozen=True)
class LabelConfig:
    heading_weight_m: float = 15.0  # cost (m-equivalent) of a fully reversed heading
    dist_cap_m: float = 200.0  # robust cap so a single outlier cannot dominate the alignment
    boundary_margin_m: float = 1.0
    boundary_window: int = 2
    chunk_points: int = 256


def align_track_to_route(
    px: np.ndarray,
    py: np.ndarray,
    heading_deg: np.ndarray,
    heading_valid: np.ndarray,
    geom: ArcGeometry,
    route,
    config: LabelConfig = LabelConfig(),
) -> Dict[str, np.ndarray]:
    """Monotone DP alignment. Returns per-point arrays: route_pos, true_arc_id, align_dist_m, align_t, align_cost."""
    route = np.asarray(route, dtype=np.int64)
    n, k = len(px), len(route)
    if n == 0 or k == 0:
        raise ValueError("align_track_to_route needs at least one point and one route arc")

    rux, ruy, rvx, rvy = geom.ux[route], geom.uy[route], geom.vx[route], geom.vy[route]
    rbear = geom.bearing_deg[route]
    hv = np.asarray(heading_valid).astype(bool) & ~np.isnan(heading_deg)
    hdg = np.where(hv, heading_deg, 0.0)

    back = np.empty((n, k), dtype=np.int16 if k < np.iinfo(np.int16).max else np.int32)
    ar = np.arange(k)
    prev: Optional[np.ndarray] = None
    for c0 in range(0, n, config.chunk_points):
        c1 = min(n, c0 + config.chunk_points)
        cost = _cost_block(px[c0:c1], py[c0:c1], hdg[c0:c1], hv[c0:c1], rux, ruy, rvx, rvy, rbear, config)
        for r in range(c1 - c0):
            i = c0 + r
            if prev is None:
                prev = cost[r].copy()
                back[i] = 0
                continue
            m = np.minimum.accumulate(prev)
            back[i] = np.maximum.accumulate(np.where(prev == m, ar, 0))
            prev = m + cost[r]

    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    del back

    d, t, _ = point_segment_distance(px, py, rux[pos], ruy[pos], rvx[pos], rvy[pos])
    hpen = _heading_penalty(hdg, hv, rbear[pos], config)
    return {
        "route_pos": pos,
        "true_arc_id": route[pos],
        "align_dist_m": d,
        "align_t": t,
        "align_cost": np.minimum(d, config.dist_cap_m) + hpen,
    }


def _heading_penalty(hdg, hv, bearing, config: LabelConfig):
    hd = angle_diff_deg(hdg, bearing)
    return np.where(hv, config.heading_weight_m * (1.0 - np.cos(np.radians(hd))) / 2.0, 0.0)


def _cost_block(px, py, hdg, hv, rux, ruy, rvx, rvy, rbear, config: LabelConfig):
    d, _, _ = point_segment_distance(px[:, None], py[:, None], rux[None, :], ruy[None, :], rvx[None, :], rvy[None, :])
    return np.minimum(d, config.dist_cap_m) + _heading_penalty(hdg[:, None], hv[:, None], rbear[None, :], config)


def boundary_margin(
    px, py, heading_deg, heading_valid, geom: ArcGeometry, route, alignment: Dict[str, np.ndarray],
    config: LabelConfig = LabelConfig(),
) -> np.ndarray:
    """
    Cost gap between the assigned route arc and the best *different* arc within
    +-boundary_window route positions (multi-edge twins with identical endpoints
    are excluded; they are handled by ``label_equiv``). Small or negative values
    mean the time-alignment boundary is uncertain at this point.
    """
    route = np.asarray(route, dtype=np.int64)
    k = len(route)
    pos = alignment["route_pos"]
    own = alignment["align_cost"]
    true_arc = alignment["true_arc_id"]
    hv = np.asarray(heading_valid).astype(bool) & ~np.isnan(heading_deg)
    hdg = np.where(hv, heading_deg, 0.0)
    best_alt = np.full(len(px), np.inf)
    for o in range(-config.boundary_window, config.boundary_window + 1):
        if o == 0:
            continue
        q = pos + o
        ok = (q >= 0) & (q < k)
        qc = np.clip(q, 0, k - 1)
        alt = route[qc]
        same_geom = (geom.from_node[alt] == geom.from_node[true_arc]) & (geom.to_node[alt] == geom.to_node[true_arc])
        ok &= ~same_geom
        d, _, _ = point_segment_distance(px, py, geom.ux[alt], geom.uy[alt], geom.vx[alt], geom.vy[alt])
        c = np.minimum(d, config.dist_cap_m) + _heading_penalty(hdg, hv, geom.bearing_deg[alt], config)
        best_alt = np.where(ok, np.minimum(best_alt, c), best_alt)
    return best_alt - own


def label_candidates(
    cands: pd.DataFrame,
    alignment: Dict[str, np.ndarray],
    margin: np.ndarray,
    geom: ArcGeometry,
    config: LabelConfig = LabelConfig(),
):
    """
    Attach labels to candidate rows.

    Returns (cand_labels_df, point_labels_df):
        cand_labels_df: label (int8: candidate == aligned arc),
                        label_equiv (int8: multi-edge twin of aligned arc - same nodes, other id),
                        label_status (categorical, per point)
        point_labels_df: per GPS point status and alignment diagnostics.
    """
    n_points = alignment["route_pos"].shape[0]
    pi = cands["point_idx"].to_numpy(np.int64)
    ai = cands["arc_id"].to_numpy(np.int64)
    true_arc = alignment["true_arc_id"]

    label = ai == true_arc[pi]
    ta = true_arc[pi]
    equiv = (~label) & (geom.from_node[ai] == geom.from_node[ta]) & (geom.to_node[ai] == geom.to_node[ta])

    has_pos = np.zeros(n_points, dtype=bool)
    has_pos[pi[label]] = True
    status = np.where(~has_pos, "off_route", np.where(margin < config.boundary_margin_m, "boundary", "ok"))
    status_cat = pd.Categorical(status, categories=LABEL_STATUSES)

    cand_labels = pd.DataFrame(
        {
            "label": label.astype(np.int8),
            "label_equiv": equiv.astype(np.int8),
            "label_status": pd.Categorical(status[pi], categories=LABEL_STATUSES),
        },
        index=cands.index,
    )
    point_labels = pd.DataFrame(
        {
            "route_pos": alignment["route_pos"].astype(np.int32),
            "true_arc_id": true_arc.astype(np.int32),
            "align_dist_m": alignment["align_dist_m"].astype(np.float32),
            "align_margin_m": np.where(np.isfinite(margin), margin, np.nan).astype(np.float32),
            "has_positive": has_pos.astype(np.int8),
            "label_status": status_cat,
        }
    )
    return cand_labels, point_labels


def label_quality(
    point_labels: pd.DataFrame,
    geom: ArcGeometry,
    route,
    px: np.ndarray,
    py: np.ndarray,
    ts: np.ndarray,
    align_t: np.ndarray,
    segment_id: np.ndarray,
    radius_m: float,
) -> Dict[str, float]:
    """Label diagnostics that do not need a model: coverage, progress consistency, ambiguity."""
    route = np.asarray(route, dtype=np.int64)
    k = len(route)
    pos = point_labels["route_pos"].to_numpy(np.int64)
    status = point_labels["label_status"].astype(str).to_numpy()
    d = point_labels["align_dist_m"].to_numpy(np.float64)
    n = len(pos)
    q = {}
    q["n_points"] = int(n)
    for s in LABEL_STATUSES:
        q[f"n_{s}"] = int((status == s).sum())
    off = status == "off_route"
    q["off_route_at_route_ends"] = int((off & ((pos == 0) | (pos == k - 1))).sum())
    q["off_route_mid_route"] = int((off & (pos > 0) & (pos < k - 1)).sum())
    on = ~off
    if on.any():
        q["align_dist_median_m"] = float(np.median(d[on]))
        q["align_dist_p95_m"] = float(np.percentile(d[on], 95))
        q["align_dist_max_m"] = float(d[on].max())
    # Points aligned to the first/last route arc but off-network lie outside the span the
    # route describes (e.g. trip 54 keeps logging while parked after the route ends). Report
    # miss rates with and without them, so the generator is not blamed for missing route data.
    route_end_off = off & ((pos == 0) | (pos == k - 1))
    for r in (30.0, 40.0, 50.0):
        if r <= radius_m + 1e-9:
            q[f"miss_rate_r{int(r)}"] = float((d > r).mean())
            keep = ~route_end_off
            q[f"miss_rate_r{int(r)}_excl_route_ends"] = float((d[keep] > r).mean()) if keep.any() else 0.0

    visited = np.zeros(k, dtype=bool)
    visited[pos[on]] = True
    span = (pos[on].min(), pos[on].max()) if on.any() else (0, -1)
    in_span = np.zeros(k, dtype=bool)
    in_span[span[0]:span[1] + 1] = True
    q["route_len"] = int(k)
    q["route_positions_visited"] = int(visited.sum())
    q["route_coverage_in_span"] = float(visited[in_span].mean()) if in_span.any() else 0.0
    unvisited_len = geom.length_m[route[in_span & ~visited]]
    q["unvisited_arc_len_median_m"] = float(np.median(unvisited_len)) if unvisited_len.size else 0.0
    q["unvisited_arc_len_max_m"] = float(unvisited_len.max()) if unvisited_len.size else 0.0

    # Route-progress consistency: distance advanced along the route between consecutive
    # labelled points should match the GPS step (same segment, both points on-route).
    cum = np.r_[0.0, np.cumsum(geom.length_m[route])]
    s_along = cum[pos] + align_t * geom.length_m[route[pos]]
    step = np.hypot(np.diff(px), np.diff(py))
    ds = np.diff(s_along)
    ok = on[1:] & on[:-1] & (segment_id[1:] == segment_id[:-1])
    if ok.any():
        err = np.abs(ds[ok] - step[ok])
        q["progress_abs_err_median_m"] = float(np.median(err))
        q["progress_abs_err_p95_m"] = float(np.percentile(err, 95))
        q["progress_err_gt25m_frac"] = float((err > 25.0).mean())
        q["backward_steps_gt5m_frac"] = float((ds[ok] < -5.0).mean())
    return q
