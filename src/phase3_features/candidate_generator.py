"""
PathSnap Phase 3 - Spatial candidate generation.

For every GPS observation we return every road arc (trip-local ``arc_id``)
whose straight-line segment lies within ``radius_m`` metres of the point.

Why not a KDTree on arc midpoints?
    Arcs are straight node-to-node segments and some are > 1.7 km long, so a
    point can be 5 m from an arc whose midpoint is 800 m away. A midpoint index
    silently misses such arcs.

Method (exact, no misses by construction):
    1. Each arc is split into ``n = ceil(L / s)`` equal sub-intervals and the
       centre of each sub-interval is indexed (``s`` = ``sample_spacing_m``).
       Every point on the arc is then within ``s / 2`` of some sample.
    2. A lossless corridor pre-filter drops samples that are not in, or next
       to, a coarse grid cell occupied by the track. This matters for trips whose
       network is cropped very generously (up to 307k arcs / 12,000 km).
    3. A dual-tree ``cKDTree.sparse_distance_matrix`` query with radius
       ``R + s/2`` returns (point, sample) pairs. Triangle inequality guarantees
       every arc within ``R`` has at least one sample within ``R + s/2``.
    4. Pairs are reduced to unique (point, arc) and the exact clamped
       point-to-segment distance is computed. Pairs with distance > R are dropped.

Coordinates: local equirectangular metres from ``pathsnap_loader`` (origin at
the median node of the trip). The projection's scale error over a trip's extent
is measured in ``scripts/benchmark_phase3.py`` against haversine.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

RECOMMENDED_RADIUS_RANGE_M = (30.0, 50.0)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CandidateConfig:
    radius_m: float = 50.0
    sample_spacing_m: float = 10.0
    max_candidates: Optional[int] = None  # keep only the k nearest per point (None = keep all)
    corridor_cell_m: float = 250.0  # coarse grid size for the lossless corridor pre-filter

    def __post_init__(self) -> None:
        if not (self.radius_m > 0):
            raise ValueError(f"radius_m must be > 0, got {self.radius_m}")
        if not (self.sample_spacing_m > 0):
            raise ValueError(f"sample_spacing_m must be > 0, got {self.sample_spacing_m}")
        if self.max_candidates is not None and self.max_candidates < 1:
            raise ValueError("max_candidates must be >= 1 or None")
        # The corridor filter is lossless only if one cell covers the full query radius.
        if self.corridor_cell_m < self.query_radius_m:
            raise ValueError(
                f"corridor_cell_m ({self.corridor_cell_m}) must be >= radius_m + sample_spacing_m/2 "
                f"({self.query_radius_m})"
            )

    @property
    def query_radius_m(self) -> float:
        return self.radius_m + self.sample_spacing_m / 2.0


# --------------------------------------------------------------------------- #
# Arc geometry (vectorised view of the trip-local network)
# --------------------------------------------------------------------------- #
@dataclass
class ArcGeometry:
    """Numpy arrays indexed by trip-local arc_id (row i == arc_id i)."""

    ux: np.ndarray
    uy: np.ndarray
    vx: np.ndarray
    vy: np.ndarray
    from_node: np.ndarray
    to_node: np.ndarray
    n_nodes: int

    def __post_init__(self) -> None:
        dx = self.vx - self.ux
        dy = self.vy - self.uy
        self.length_m = np.hypot(dx, dy)
        # Compass bearing in the local metric frame: 0 = north, 90 = east.
        self.bearing_deg = np.degrees(np.arctan2(dx, dy)) % 360.0
        with np.errstate(divide="ignore", invalid="ignore"):
            self.unit_x = np.where(self.length_m > 1e-9, dx / self.length_m, 0.0)
            self.unit_y = np.where(self.length_m > 1e-9, dy / self.length_m, 0.0)

    @property
    def n_arcs(self) -> int:
        return int(self.ux.shape[0])

    @classmethod
    def from_trip(cls, trip) -> "ArcGeometry":
        a = trip.arcs_df
        # arc_id must equal the row index for O(1) lookups.
        if not np.array_equal(a["arc_id"].to_numpy(), np.arange(len(a))):
            raise ValueError("arcs_df.arc_id must be 0..n-1 in row order")
        return cls(
            ux=a["ux"].to_numpy(np.float64),
            uy=a["uy"].to_numpy(np.float64),
            vx=a["vx"].to_numpy(np.float64),
            vy=a["vy"].to_numpy(np.float64),
            from_node=a["from_node"].to_numpy(np.int64),
            to_node=a["to_node"].to_numpy(np.int64),
            n_nodes=int(len(trip.nodes_df)),
        )


# --------------------------------------------------------------------------- #
# Vectorised geometry helpers
# --------------------------------------------------------------------------- #
def point_segment_distance(px, py, ux, uy, vx, vy):
    """
    Vectorised distance from points to directed segments (all inputs broadcastable).

    Returns (distance_m, t_clamped in [0, 1], t_raw). ``t_raw`` < 0 or > 1 means the
    perpendicular foot falls outside the segment and the nearest point is an endpoint.
    """
    px, py, ux, uy, vx, vy = (np.asarray(a, dtype=np.float64) for a in (px, py, ux, uy, vx, vy))
    dx = vx - ux
    dy = vy - uy
    l2 = dx * dx + dy * dy
    num = (px - ux) * dx + (py - uy) * dy
    nonzero = l2 > 1e-12
    # zero-length arcs (self-loops / coincident nodes): distance to the single node, t = 0
    t_raw = np.divide(num, l2, out=np.zeros(np.broadcast(num, l2).shape), where=nonzero)
    t = np.clip(t_raw, 0.0, 1.0)
    qx = ux + t * dx
    qy = uy + t * dy
    return np.hypot(px - qx, py - qy), t, t_raw


def angle_diff_deg(a, b):
    """Smallest absolute difference between two compass angles, in [0, 180]."""
    d = np.abs((np.asarray(a) - np.asarray(b)) % 360.0)
    return np.minimum(d, 360.0 - d)


def densify_arcs(geom: ArcGeometry, spacing_m: float, arc_subset: Optional[np.ndarray] = None):
    """
    Sample arcs at the centres of ``ceil(L / spacing)`` equal sub-intervals.
    Every point on an arc is within ``spacing_m / 2`` of one of its samples.
    Returns (sample_x, sample_y, sample_arc_id).
    """
    arcs = np.arange(geom.n_arcs) if arc_subset is None else np.asarray(arc_subset, dtype=np.int64)
    if arcs.size == 0:
        empty = np.empty(0)
        return empty, empty, np.empty(0, dtype=np.int64)
    length = geom.length_m[arcs]
    n_sub = np.maximum(1, np.ceil(length / spacing_m)).astype(np.int64)
    sample_arc = np.repeat(arcs, n_sub)
    starts = np.repeat(np.cumsum(n_sub) - n_sub, n_sub)
    offset = np.arange(sample_arc.size, dtype=np.int64) - starts
    frac = (offset + 0.5) / np.repeat(n_sub, n_sub)
    sx = geom.ux[sample_arc] + frac * (geom.vx[sample_arc] - geom.ux[sample_arc])
    sy = geom.uy[sample_arc] + frac * (geom.vy[sample_arc] - geom.uy[sample_arc])
    return sx, sy, sample_arc


def _grid_keys(x: np.ndarray, y: np.ndarray, cell: float) -> np.ndarray:
    cx = np.floor(x / cell).astype(np.int64)
    cy = np.floor(y / cell).astype(np.int64)
    return (cx << 32) ^ (cy & 0xFFFFFFFF)


def corridor_arc_subset(geom: ArcGeometry, px: np.ndarray, py: np.ndarray, config: CandidateConfig) -> np.ndarray:
    """
    Lossless pre-filter: keep arcs whose bounding box intersects the track's bounding
    box expanded by the query radius. The finer per-sample grid filter is applied
    later in ``generate_candidates``.
    """
    r = config.query_radius_m
    minx, maxx = px.min() - r, px.max() + r
    miny, maxy = py.min() - r, py.max() + r
    ax0 = np.minimum(geom.ux, geom.vx)
    ax1 = np.maximum(geom.ux, geom.vx)
    ay0 = np.minimum(geom.uy, geom.vy)
    ay1 = np.maximum(geom.uy, geom.vy)
    keep = (ax1 >= minx) & (ax0 <= maxx) & (ay1 >= miny) & (ay0 <= maxy)
    return np.flatnonzero(keep)


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
CANDIDATE_COLUMNS = ["point_idx", "arc_id", "dist_m", "proj_t", "proj_t_raw"]


def generate_candidates(
    geom: ArcGeometry,
    px: np.ndarray,
    py: np.ndarray,
    config: CandidateConfig = CandidateConfig(),
    return_stats: bool = False,
):
    """
    Generate spatial candidates for GPS points ``(px, py)`` (local metres).

    Returns a DataFrame sorted by (point_idx, dist_m) with columns
    ``CANDIDATE_COLUMNS``. ``arc_id`` is the trip-local arc id (row index in .arcs).
    Points with no arc within the radius simply have no rows.
    """
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    n_points = px.shape[0]
    stats = {"n_points": int(n_points), "n_arcs_network": geom.n_arcs}

    if n_points == 0 or geom.n_arcs == 0:
        df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(CANDIDATE_COLUMNS, ["int32", "int32", "float32", "float32", "float32"])})
        return (df, stats) if return_stats else df

    # 1) bounding-box crop (lossless)
    arc_subset = corridor_arc_subset(geom, px, py, config)
    stats["n_arcs_bbox"] = int(arc_subset.size)

    # 2) densify the surviving arcs
    sx, sy, s_arc = densify_arcs(geom, config.sample_spacing_m, arc_subset)
    stats["n_samples_bbox"] = int(sx.size)

    # 3) corridor grid filter (lossless: cell >= query radius, so any sample within the
    #    query radius of a GPS point is in the point's cell or one of its 8 neighbours)
    cell = config.corridor_cell_m
    gx = np.floor(px / cell).astype(np.int64)
    gy = np.floor(py / cell).astype(np.int64)
    occ = set()
    for ox in (-1, 0, 1):
        for oy in (-1, 0, 1):
            occ.update(np.unique(((gx + ox) << 32) ^ ((gy + oy) & 0xFFFFFFFF)).tolist())
    occ_arr = np.fromiter(occ, dtype=np.int64, count=len(occ))
    keep = np.isin(_grid_keys(sx, sy, cell), occ_arr)
    sx, sy, s_arc = sx[keep], sy[keep], s_arc[keep]
    stats["n_samples_indexed"] = int(sx.size)

    if sx.size == 0:
        df = pd.DataFrame({c: pd.Series(dtype=t) for c, t in zip(CANDIDATE_COLUMNS, ["int32", "int32", "float32", "float32", "float32"])})
        return (df, stats) if return_stats else df

    # 4) dual-tree radius query
    gps_tree = cKDTree(np.column_stack([px, py]))
    sample_tree = cKDTree(np.column_stack([sx, sy]))
    pairs = gps_tree.sparse_distance_matrix(sample_tree, config.query_radius_m, output_type="ndarray")
    stats["n_raw_pairs"] = int(pairs.shape[0])

    p_idx = pairs["i"].astype(np.int64)
    a_idx = s_arc[pairs["j"]]
    key = np.unique(p_idx * geom.n_arcs + a_idx)
    p_idx = key // geom.n_arcs
    a_idx = key % geom.n_arcs

    # 5) exact clamped point-to-segment distance
    dist, t, t_raw = point_segment_distance(
        px[p_idx], py[p_idx], geom.ux[a_idx], geom.uy[a_idx], geom.vx[a_idx], geom.vy[a_idx]
    )
    within = dist <= config.radius_m
    p_idx, a_idx, dist, t, t_raw = p_idx[within], a_idx[within], dist[within], t[within], t_raw[within]

    # sort by (point, distance)
    order = np.lexsort((a_idx, dist, p_idx))
    p_idx, a_idx, dist, t, t_raw = p_idx[order], a_idx[order], dist[order], t[order], t_raw[order]

    if config.max_candidates is not None and p_idx.size:
        rank = _rank_within_groups(p_idx)
        sel = rank < config.max_candidates
        p_idx, a_idx, dist, t, t_raw = p_idx[sel], a_idx[sel], dist[sel], t[sel], t_raw[sel]

    df = pd.DataFrame(
        {
            "point_idx": p_idx.astype(np.int32),
            "arc_id": a_idx.astype(np.int32),
            "dist_m": dist.astype(np.float32),
            "proj_t": t.astype(np.float32),
            "proj_t_raw": t_raw.astype(np.float32),
        }
    )
    counts = np.bincount(p_idx, minlength=n_points)
    stats.update(
        n_candidates=int(df.shape[0]),
        points_without_candidates=int((counts == 0).sum()),
        cands_per_point_mean=float(counts.mean()),
        cands_per_point_median=float(np.median(counts)),
        cands_per_point_p95=float(np.percentile(counts, 95)),
        cands_per_point_max=int(counts.max()),
    )
    return (df, stats) if return_stats else df


def _rank_within_groups(sorted_group_ids: np.ndarray) -> np.ndarray:
    """0-based position of each element within its run of equal group ids (input must be grouped)."""
    n = sorted_group_ids.size
    if n == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_group_ids)) + 1]
    run_start = np.repeat(starts, np.diff(np.r_[starts, n]))
    return np.arange(n) - run_start


def brute_force_candidates(geom: ArcGeometry, px: np.ndarray, py: np.ndarray, radius_m: float) -> pd.DataFrame:
    """O(points x arcs) reference implementation - used only by tests/benchmarks on small inputs."""
    rows_p, rows_a, rows_d = [], [], []
    for i in range(len(px)):
        d, _, _ = point_segment_distance(px[i], py[i], geom.ux, geom.uy, geom.vx, geom.vy)
        hit = np.flatnonzero(d <= radius_m)
        rows_p.append(np.full(hit.size, i))
        rows_a.append(hit)
        rows_d.append(d[hit])
    df = pd.DataFrame(
        {
            "point_idx": np.concatenate(rows_p) if rows_p else np.empty(0, int),
            "arc_id": np.concatenate(rows_a) if rows_a else np.empty(0, int),
            "dist_m": np.concatenate(rows_d) if rows_d else np.empty(0),
        }
    )
    return df.sort_values(["point_idx", "dist_m", "arc_id"]).reset_index(drop=True)


def generate_trip_candidates(trip, config: CandidateConfig = CandidateConfig(), return_stats: bool = False):
    """Convenience wrapper: candidates for a loaded ``TripData``."""
    geom = ArcGeometry.from_trip(trip)
    px = trip.track_df["x_m"].to_numpy(np.float64)
    py = trip.track_df["y_m"].to_numpy(np.float64)
    out = generate_candidates(geom, px, py, config, return_stats=return_stats)
    return out
