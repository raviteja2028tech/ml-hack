"""
PathSnap Phase 3 - Candidate feature engineering.

LEAKAGE CONTRACT
    Everything in this module is computed from the GPS track (.track) and the
    trip-local road network (.nodes/.arcs) only. It never reads ``trip.route``
    and never imports ``route_labeler``. ``tests/test_phase3.py`` checks this by
    recomputing features with the route removed and asserting bit-identical
    output. No statistic is fitted across trips here: any scaler, imputer or
    encoder belongs to Phase 4 and must be fitted on training folds only.

Inference setting: map matching is done offline on the whole trip, so features
may use neighbouring GPS points (i-1, i+1, ...) and their spatial candidates.
They never use the label of any point.

Feature groups (see FEATURE_COLUMNS):
    distance     - GPS-to-road distance, rank and gap to the nearest candidate
    heading      - GPS heading vs arc bearing (directed and undirected)
    segment      - arc length, projection position along the arc
    speed        - vehicle speed / acceleration / along- and cross-road velocity
    connectivity - node degrees, two-way and multi-edge flags, hop distance
                   from/to the neighbouring points' best candidates
    temporal     - candidate persistence and fit to a +-2 point trajectory window
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.candidate_generator import ArcGeometry, angle_diff_deg, point_segment_distance

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FeatureConfig:
    max_gap_s: float = 30.0  # a time gap above this splits the track into segments
    min_heading_disp_m: float = 3.0  # centred displacement needed for a measured heading
    smooth_window: int = 5  # rolling-median window (points) for speed smoothing
    persistence_halfwindow: int = 5  # +-points used for candidate persistence
    trajectory_halfwindow: int = 2  # +-points used for trajectory-fit distance
    max_hops: int = 3  # BFS depth for hop-distance features (value max_hops+1 = farther/unknown)
    anchor_heading_weight_m: float = 7.5  # anchor score = dist + w * (1 - heading_cos)


ID_COLUMNS = ["trip_id", "point_idx", "timestamp", "segment_id", "arc_id"]

FEATURE_COLUMNS = [
    # distance
    "dist_m", "dist_rank", "dist_gap_to_min_m", "dist_ratio_to_min", "n_candidates", "min_dist_point_m",
    # segment / projection
    "arc_length_m", "proj_t", "proj_beyond_end_m",
    # heading
    "heading_diff_deg", "heading_cos", "heading_diff_undirected_deg", "heading_valid",
    # speed / kinematics
    "speed_mps", "speed_smooth_mps", "accel_mps2", "along_speed_mps", "cross_speed_mps",
    "turn_rate_deg_s", "dt_prev_s", "dt_next_s", "step_prev_m", "step_next_m",
    # connectivity
    "from_in_degree", "to_out_degree", "is_two_way", "has_multi_edge", "is_self_loop",
    "in_prev_cands", "in_next_cands", "pred_in_prev_cands", "succ_in_next_cands",
    "hops_from_prev_anchor", "hops_to_next_anchor", "is_prev_anchor", "is_next_anchor",
    # temporal consistency
    "persistence_ratio", "win_mean_dist_m", "win_max_dist_m", "win_heading_cos_mean",
    "has_prev_point", "has_next_point",
]

_INT8_FEATURES = {
    "heading_valid", "is_two_way", "has_multi_edge", "is_self_loop", "in_prev_cands", "in_next_cands",
    "pred_in_prev_cands", "succ_in_next_cands", "hops_from_prev_anchor", "hops_to_next_anchor",
    "is_prev_anchor", "is_next_anchor", "has_prev_point", "has_next_point",
}
_INT16_FEATURES = {"dist_rank", "n_candidates", "from_in_degree", "to_out_degree"}
FEATURE_DTYPES = {
    c: (np.int8 if c in _INT8_FEATURES else np.int16 if c in _INT16_FEATURES else np.float32)
    for c in FEATURE_COLUMNS
}


# --------------------------------------------------------------------------- #
# GPS kinematics (point level)
# --------------------------------------------------------------------------- #
def compute_point_kinematics(x: np.ndarray, y: np.ndarray, ts: np.ndarray, config: FeatureConfig = FeatureConfig()) -> pd.DataFrame:
    """
    Per-point kinematics from positions (metres) and timestamps (s).

    The track is split into segments at time gaps > ``max_gap_s``; no difference or
    window crosses a segment boundary. Heading is the centred displacement direction
    (i-1 -> i+1). When that displacement is < ``min_heading_disp_m`` (stopped or
    crawling) the heading is unreliable: we carry the last measured heading forward
    within the segment (a stopped vehicle keeps its orientation) and set
    ``heading_valid = 0``.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ts = np.asarray(ts, dtype=np.float64)
    n = x.size
    idx = np.arange(n)

    dt_raw = np.r_[np.nan, np.diff(ts)]
    gap_before = np.zeros(n, dtype=bool)
    gap_before[1:] = dt_raw[1:] > config.max_gap_s
    segment_id = np.cumsum(gap_before).astype(np.int32)
    first = np.r_[True, segment_id[1:] != segment_id[:-1]] if n else np.zeros(0, bool)
    last = np.r_[segment_id[1:] != segment_id[:-1], True] if n else np.zeros(0, bool)

    step = np.r_[np.nan, np.hypot(np.diff(x), np.diff(y))]
    dt_prev = np.where(first, np.nan, dt_raw)
    step_prev = np.where(first, np.nan, step)
    dt_next = np.r_[dt_prev[1:], np.nan]
    step_next = np.r_[step_prev[1:], np.nan]

    ip = np.where(first, idx, idx - 1)
    inx = np.where(last, idx, idx + 1)
    cdx = x[inx] - x[ip]
    cdy = y[inx] - y[ip]
    cdt = ts[inx] - ts[ip]
    disp = np.hypot(cdx, cdy)
    with np.errstate(divide="ignore", invalid="ignore"):
        speed = np.where(cdt > 0, disp / cdt, 0.0)
        vel_x = np.where(cdt > 0, cdx / cdt, 0.0)
        vel_y = np.where(cdt > 0, cdy / cdt, 0.0)

    heading_valid = disp >= config.min_heading_disp_m
    heading_raw = np.where(heading_valid, np.degrees(np.arctan2(cdx, cdy)) % 360.0, np.nan)

    df = pd.DataFrame({"segment_id": segment_id, "heading": heading_raw, "speed": speed})
    g = df.groupby("segment_id", sort=False)
    heading = g["heading"].transform(lambda s: s.ffill().bfill()).to_numpy()
    speed_smooth = (
        g["speed"].transform(lambda s: s.rolling(config.smooth_window, center=True, min_periods=1).median()).to_numpy()
    )

    # acceleration: centred difference of smoothed speed over centred dt
    with np.errstate(divide="ignore", invalid="ignore"):
        accel = np.where(cdt > 0, (speed_smooth[inx] - speed_smooth[ip]) / cdt, 0.0)
        # turn rate between consecutive filled headings
        dh = angle_diff_deg(heading, np.r_[heading[:1], heading[:-1]])
        turn_rate = np.where(first | np.isnan(dh), 0.0, dh / np.where(np.isnan(dt_prev), 1.0, dt_prev))

    return pd.DataFrame(
        {
            "segment_id": segment_id,
            "has_prev_point": (~first).astype(np.int8),
            "has_next_point": (~last).astype(np.int8),
            "dt_prev_s": dt_prev,
            "dt_next_s": dt_next,
            "step_prev_m": step_prev,
            "step_next_m": step_next,
            "speed_mps": speed,
            "speed_smooth_mps": speed_smooth,
            "accel_mps2": accel,
            "vel_x": vel_x,
            "vel_y": vel_y,
            "heading_deg": heading,  # NaN only if a whole segment never moved
            "heading_valid": heading_valid.astype(np.int8),
            "turn_rate_deg_s": turn_rate,
        }
    )


# --------------------------------------------------------------------------- #
# Road-network graph (arc level)
# --------------------------------------------------------------------------- #
@dataclass
class ArcGraph:
    out_ptr: np.ndarray  # node -> slice into out_arcs
    out_arcs: np.ndarray  # arcs sorted by from_node
    in_ptr: np.ndarray
    in_arcs: np.ndarray  # arcs sorted by to_node
    from_in_degree: np.ndarray  # per arc: in-degree of its from_node
    to_out_degree: np.ndarray  # per arc: out-degree of its to_node
    is_two_way: np.ndarray  # reverse arc (to->from) exists
    has_multi_edge: np.ndarray  # another arc has identical (from, to)
    is_self_loop: np.ndarray

    def successors(self, arc: int, geom: ArcGeometry) -> np.ndarray:
        n = geom.to_node[arc]
        return self.out_arcs[self.out_ptr[n]:self.out_ptr[n + 1]]

    def predecessors(self, arc: int, geom: ArcGeometry) -> np.ndarray:
        n = geom.from_node[arc]
        return self.in_arcs[self.in_ptr[n]:self.in_ptr[n + 1]]


def build_arc_graph(geom: ArcGeometry) -> ArcGraph:
    nn = geom.n_nodes
    fr, to = geom.from_node, geom.to_node
    out_arcs = np.argsort(fr, kind="stable")
    out_ptr = np.r_[0, np.cumsum(np.bincount(fr, minlength=nn))]
    in_arcs = np.argsort(to, kind="stable")
    in_ptr = np.r_[0, np.cumsum(np.bincount(to, minlength=nn))]
    out_deg = np.diff(out_ptr)
    in_deg = np.diff(in_ptr)

    key = fr * np.int64(nn) + to
    rev_key = to * np.int64(nn) + fr
    uniq, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    return ArcGraph(
        out_ptr=out_ptr,
        out_arcs=out_arcs,
        in_ptr=in_ptr,
        in_arcs=in_arcs,
        from_in_degree=in_deg[fr],
        to_out_degree=out_deg[to],
        is_two_way=np.isin(rev_key, uniq) & (fr != to),
        has_multi_edge=cnt[inv] > 1,
        is_self_loop=fr == to,
    )


def _bfs_hops(start: int, graph: ArcGraph, geom: ArcGeometry, max_hops: int, forward: bool) -> Dict[int, int]:
    """Hop distance from ``start`` over the arc-successor graph (or predecessor graph)."""
    hops = {int(start): 0}
    q = deque([int(start)])
    while q:
        a = q.popleft()
        h = hops[a]
        if h >= max_hops:
            continue
        nbrs = graph.successors(a, geom) if forward else graph.predecessors(a, geom)
        for b in nbrs.tolist():
            if b not in hops:
                hops[b] = h + 1
                q.append(b)
    return hops


# --------------------------------------------------------------------------- #
# Candidate features
# --------------------------------------------------------------------------- #
def _segment_keys(point_idx: np.ndarray, arc: np.ndarray, n_arcs: int) -> np.ndarray:
    return point_idx.astype(np.int64) * np.int64(n_arcs) + arc.astype(np.int64)


def build_candidate_features(
    cands: pd.DataFrame,
    kin: pd.DataFrame,
    geom: ArcGeometry,
    graph: ArcGraph,
    px: np.ndarray,
    py: np.ndarray,
    config: FeatureConfig = FeatureConfig(),
) -> pd.DataFrame:
    """
    Compute FEATURE_COLUMNS for every (point, candidate arc) row.

    ``cands`` must come from ``generate_candidates`` (sorted by point_idx, dist_m).
    Returns a new DataFrame aligned row-for-row with ``cands``.
    """
    m = len(cands)
    n_points = len(kin)
    out: Dict[str, np.ndarray] = {}
    if m == 0:
        return pd.DataFrame({c: pd.Series(dtype="float32") for c in FEATURE_COLUMNS})

    pi = cands["point_idx"].to_numpy(np.int64)
    ai = cands["arc_id"].to_numpy(np.int64)
    dist = cands["dist_m"].to_numpy(np.float64)
    seg = kin["segment_id"].to_numpy()
    has_prev = kin["has_prev_point"].to_numpy().astype(bool)
    has_next = kin["has_next_point"].to_numpy().astype(bool)

    # ---- distance group ---------------------------------------------------- #
    starts = np.r_[0, np.flatnonzero(np.diff(pi)) + 1]
    run_len = np.diff(np.r_[starts, m])
    min_dist = np.repeat(np.minimum.reduceat(dist, starts), run_len)
    out["dist_m"] = dist
    out["dist_rank"] = np.arange(m) - np.repeat(starts, run_len)
    out["dist_gap_to_min_m"] = dist - min_dist
    out["dist_ratio_to_min"] = (dist + 1.0) / (min_dist + 1.0)
    out["n_candidates"] = np.repeat(run_len, run_len)
    out["min_dist_point_m"] = min_dist

    # ---- segment group ------------------------------------------------------ #
    length = geom.length_m[ai]
    t_raw = cands["proj_t_raw"].to_numpy(np.float64)
    out["arc_length_m"] = length
    out["proj_t"] = cands["proj_t"].to_numpy(np.float64)
    out["proj_beyond_end_m"] = np.maximum(0.0, np.maximum(-t_raw, t_raw - 1.0)) * length

    # ---- heading group ------------------------------------------------------ #
    heading = kin["heading_deg"].to_numpy()[pi]
    bearing = geom.bearing_deg[ai]
    hd = angle_diff_deg(heading, bearing)
    no_heading = np.isnan(hd)
    hd = np.where(no_heading, 90.0, hd)  # neutral when a segment never moved (constant, not fitted)
    hcos = np.cos(np.radians(hd))
    out["heading_diff_deg"] = hd
    out["heading_cos"] = hcos
    out["heading_diff_undirected_deg"] = np.minimum(hd, 180.0 - hd)
    out["heading_valid"] = kin["heading_valid"].to_numpy()[pi]

    # ---- speed group -------------------------------------------------------- #
    for c in ("speed_mps", "speed_smooth_mps", "accel_mps2", "turn_rate_deg_s"):
        out[c] = kin[c].to_numpy()[pi]
    vx = kin["vel_x"].to_numpy()[pi]
    vy = kin["vel_y"].to_numpy()[pi]
    out["along_speed_mps"] = vx * geom.unit_x[ai] + vy * geom.unit_y[ai]
    out["cross_speed_mps"] = np.abs(vx * geom.unit_y[ai] - vy * geom.unit_x[ai])
    # NaN at segment boundaries -> 0 / -1 sentinels (constants, not fitted); flags carry the info
    out["dt_prev_s"] = np.nan_to_num(kin["dt_prev_s"].to_numpy()[pi], nan=-1.0)
    out["dt_next_s"] = np.nan_to_num(kin["dt_next_s"].to_numpy()[pi], nan=-1.0)
    out["step_prev_m"] = np.nan_to_num(kin["step_prev_m"].to_numpy()[pi], nan=-1.0)
    out["step_next_m"] = np.nan_to_num(kin["step_next_m"].to_numpy()[pi], nan=-1.0)

    # ---- static connectivity ------------------------------------------------ #
    out["from_in_degree"] = graph.from_in_degree[ai]
    out["to_out_degree"] = graph.to_out_degree[ai]
    out["is_two_way"] = graph.is_two_way[ai].astype(np.int8)
    out["has_multi_edge"] = graph.has_multi_edge[ai].astype(np.int8)
    out["is_self_loop"] = graph.is_self_loop[ai].astype(np.int8)

    # ---- dynamic connectivity w.r.t. neighbouring points' candidates ------- #
    na = geom.n_arcs
    nn = np.int64(geom.n_nodes)
    keys_sorted = np.sort(_segment_keys(pi, ai, na))

    def _member(keys: np.ndarray, valid: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(keys_sorted, keys)
        pos = np.minimum(pos, keys_sorted.size - 1)
        return valid & (keys_sorted[pos] == keys)

    prev_ok = has_prev[pi]
    next_ok = has_next[pi]
    out["in_prev_cands"] = _member(_segment_keys(pi - 1, ai, na), prev_ok).astype(np.int8)
    out["in_next_cands"] = _member(_segment_keys(pi + 1, ai, na), next_ok).astype(np.int8)

    # predecessor of this arc among the previous point's candidates:
    # exists c' at i-1 with to_node[c'] == from_node[a]
    prev_to_keys = np.unique((pi + 1) * nn + geom.to_node[ai])  # what point i+1 can reach from i's candidates
    cur_from_keys = pi * nn + geom.from_node[ai]
    pos = np.minimum(np.searchsorted(prev_to_keys, cur_from_keys), prev_to_keys.size - 1)
    out["pred_in_prev_cands"] = (prev_ok & (prev_to_keys[pos] == cur_from_keys)).astype(np.int8)
    next_from_keys = np.unique((pi - 1) * nn + geom.from_node[ai])
    cur_to_keys = pi * nn + geom.to_node[ai]
    pos = np.minimum(np.searchsorted(next_from_keys, cur_to_keys), next_from_keys.size - 1)
    out["succ_in_next_cands"] = (next_ok & (next_from_keys[pos] == cur_to_keys)).astype(np.int8)

    # anchor = label-free "best guess" per point (distance + heading), used for hop features
    score = dist + config.anchor_heading_weight_m * (1.0 - hcos)
    order = np.lexsort((score, pi))
    first_of_point = order[np.r_[0, np.flatnonzero(np.diff(pi[order])) + 1]]
    anchor = np.full(n_points, -1, dtype=np.int64)
    anchor[pi[first_of_point]] = ai[first_of_point]

    far = config.max_hops + 1
    hops_prev = np.full(m, far, dtype=np.int8)
    hops_next = np.full(m, far, dtype=np.int8)
    fwd_cache: Dict[int, Dict[int, int]] = {}
    bwd_cache: Dict[int, Dict[int, int]] = {}
    for s, ln in zip(starts.tolist(), run_len.tolist()):
        p = int(pi[s])
        arcs_here = ai[s:s + ln].tolist()
        if has_prev[p] and anchor[p - 1] >= 0:
            a0 = int(anchor[p - 1])
            h = fwd_cache.get(a0)
            if h is None:
                h = fwd_cache[a0] = _bfs_hops(a0, graph, geom, config.max_hops, forward=True)
            hops_prev[s:s + ln] = [h.get(a, far) for a in arcs_here]
        if has_next[p] and anchor[p + 1] >= 0:
            a1 = int(anchor[p + 1])
            h = bwd_cache.get(a1)
            if h is None:
                h = bwd_cache[a1] = _bfs_hops(a1, graph, geom, config.max_hops, forward=False)
            hops_next[s:s + ln] = [h.get(a, far) for a in arcs_here]
    out["hops_from_prev_anchor"] = hops_prev
    out["hops_to_next_anchor"] = hops_next
    out["is_prev_anchor"] = (hops_prev == 0).astype(np.int8)
    out["is_next_anchor"] = (hops_next == 0).astype(np.int8)

    # ---- temporal consistency ---------------------------------------------- #
    w = config.persistence_halfwindow
    pres = np.zeros(m, dtype=np.float64)
    possible = np.zeros(m, dtype=np.float64)
    for o in range(-w, w + 1):
        if o == 0:
            continue
        q = pi + o
        inside = (q >= 0) & (q < n_points)
        same_seg = np.zeros(m, dtype=bool)
        same_seg[inside] = seg[q[inside]] == seg[pi[inside]]
        possible += same_seg
        pres += _member(_segment_keys(q, ai, na), same_seg)
    out["persistence_ratio"] = np.where(possible > 0, pres / np.maximum(possible, 1), 0.0)

    hw = config.trajectory_halfwindow
    hdg_all = kin["heading_deg"].to_numpy()
    d_sum = dist.copy()
    d_max = dist.copy()
    cnt = np.ones(m)
    hc_sum = hcos.copy()
    for o in range(-hw, hw + 1):
        if o == 0:
            continue
        q = pi + o
        inside = (q >= 0) & (q < n_points)
        qc = np.clip(q, 0, n_points - 1)
        ok = inside & (seg[qc] == seg[pi])
        d_o, _, _ = point_segment_distance(px[qc], py[qc], geom.ux[ai], geom.uy[ai], geom.vx[ai], geom.vy[ai])
        h_o = angle_diff_deg(hdg_all[qc], bearing)
        h_o = np.cos(np.radians(np.where(np.isnan(h_o), 90.0, h_o)))
        d_sum += np.where(ok, d_o, 0.0)
        d_max = np.where(ok, np.maximum(d_max, d_o), d_max)
        hc_sum += np.where(ok, h_o, 0.0)
        cnt += ok
    out["win_mean_dist_m"] = d_sum / cnt
    out["win_max_dist_m"] = d_max
    out["win_heading_cos_mean"] = hc_sum / cnt
    out["has_prev_point"] = has_prev[pi].astype(np.int8)
    out["has_next_point"] = has_next[pi].astype(np.int8)

    feats = pd.DataFrame(out, index=cands.index)[FEATURE_COLUMNS]
    # Fixed, trip-independent dtypes (stable Parquet schema, ~half the RAM of float64)
    return feats.astype(FEATURE_DTYPES)


def build_trip_features(trip, cands: pd.DataFrame, geom: Optional[ArcGeometry] = None,
                        graph: Optional[ArcGraph] = None, config: FeatureConfig = FeatureConfig()):
    """Convenience wrapper for a loaded TripData. Returns (features_df, kinematics_df)."""
    geom = geom or ArcGeometry.from_trip(trip)
    graph = graph or build_arc_graph(geom)
    tr = trip.track_df
    px = tr["x_m"].to_numpy(np.float64)
    py = tr["y_m"].to_numpy(np.float64)
    kin = compute_point_kinematics(px, py, tr["timestamp"].to_numpy(np.float64), config)
    feats = build_candidate_features(cands, kin, geom, graph, px, py, config)
    return feats, kin
