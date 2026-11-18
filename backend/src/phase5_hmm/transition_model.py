"""
HMM transition model for PathSnap.

A transition goes from candidate ``a`` (projection ``t_a`` on arc ``a``) at GPS point
``k-1`` to candidate ``b`` (``t_b`` on arc ``b``) at point ``k``. Its log-score follows
Newson & Krumm (2009), extended with speed and direction checks:

    d_route = (1 - t_a)*len(a) + shortest_path(to(a) -> from(b)) + t_b*len(b)   (a != b)
            = |t_b - t_a| * len(a)                                           (same arc / parallel twin)
    log p   = -|d_route - d_gps| / beta
              - speed_penalty * max(0, d_route/dt - v_max)
              - backward_penalty * max(0, backward_move - tol)                (same arc, moving backwards)
              - uturn_penalty                                                (b is the reverse twin of a)
    unreachable within the search radius -> ``disconnected_logp`` (finite, so the
    decoder can still recover from map gaps instead of failing).

Shortest paths use a bounded Dijkstra on the trip's directed node graph. Results are
cached per source node with the radius they were computed for.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np


@dataclass(frozen=True)
class TransitionConfig:
    beta_m: float = 10.0  # scale of |d_route - d_gps| (metres per nat)
    max_speed_mps: float = 45.0  # ~160 km/h: faster implied travel is implausible
    speed_penalty_per_mps: float = 0.5
    backward_tol_m: float = 5.0  # GPS jitter tolerated backwards along one arc
    backward_penalty_per_m: float = 0.2
    uturn_penalty: float = 3.0
    disconnected_logp: float = -50.0
    search_factor: float = 3.0  # Dijkstra radius = factor*d_gps + slack
    search_slack_m: float = 250.0
    max_search_m: float = 4000.0

    def to_dict(self) -> Dict:
        return asdict(self)


class RoadGraph:
    """Directed node graph of one trip's road network (trip-local node/arc ids)."""

    def __init__(self, from_node: Sequence[int], to_node: Sequence[int], length_m: Sequence[float], n_nodes: int):
        self.from_node = np.asarray(from_node, dtype=np.int64)
        self.to_node = np.asarray(to_node, dtype=np.int64)
        self.length_m = np.asarray(length_m, dtype=np.float64)
        self.n_nodes = int(max(n_nodes, self.from_node.max(initial=-1) + 1, self.to_node.max(initial=-1) + 1))
        order = np.argsort(self.from_node, kind="stable")
        counts = np.bincount(self.from_node, minlength=self.n_nodes)
        self._indptr: List[int] = np.r_[0, np.cumsum(counts)].tolist()
        self._nbr: List[int] = self.to_node[order].tolist()
        self._w: List[float] = self.length_m[order].tolist()
        # python lists for fast scalar access inside the decoder loops
        self.from_list: List[int] = self.from_node.tolist()
        self.to_list: List[int] = self.to_node.tolist()
        self.len_list: List[float] = self.length_m.tolist()
        self._cache: Dict[int, tuple] = {}
        self.n_dijkstra = 0

    @classmethod
    def from_geometry(cls, geom) -> "RoadGraph":
        return cls(geom.from_node, geom.to_node, geom.length_m, geom.n_nodes)

    def distances_from(self, source: int, limit_m: float) -> Dict[int, float]:
        """Shortest network distance from ``source`` to every node within ``limit_m`` (cached)."""
        hit = self._cache.get(source)
        if hit is not None and hit[0] >= limit_m:
            return hit[1]
        lim = 256.0
        while lim < limit_m:  # power-of-two radius buckets -> good cache reuse
            lim *= 2.0
        indptr, nbr, w = self._indptr, self._nbr, self._w
        dist = {source: 0.0}
        heap = [(0.0, source)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist[u]:
                continue
            for k in range(indptr[u], indptr[u + 1]):
                nd = d + w[k]
                if nd <= lim:
                    v = nbr[k]
                    if nd < dist.get(v, math.inf):
                        dist[v] = nd
                        heapq.heappush(heap, (nd, v))
        self._cache[source] = (lim, dist)
        self.n_dijkstra += 1
        return dist


def transition_logp(
    graph: RoadGraph,
    arcs_a: Sequence[int],
    t_a: Sequence[float],
    arcs_b: Sequence[int],
    t_b: Sequence[float],
    d_gps: float,
    dt: float,
    cfg: TransitionConfig,
) -> np.ndarray:
    """Log transition matrix of shape (len(arcs_a), len(arcs_b))."""
    fr, to, ln = graph.from_list, graph.to_list, graph.len_list
    limit = min(cfg.max_search_m, cfg.search_factor * d_gps + cfg.search_slack_m)
    ka, kb = len(arcs_a), len(arcs_b)
    d_route = np.full((ka, kb), np.inf)
    extra = np.zeros((ka, kb))
    for i in range(ka):
        a = int(arcs_a[i])
        ta = float(t_a[i])
        fa, ta_node, la = fr[a], to[a], ln[a]
        reach = None
        for j in range(kb):
            b = int(arcs_b[j])
            tb = float(t_b[j])
            if a == b or (fr[b] == fa and to[b] == ta_node):  # same arc or parallel duplicate arc
                move = (tb - ta) * la
                d_route[i, j] = abs(move)
                if move < -cfg.backward_tol_m:
                    extra[i, j] += cfg.backward_penalty_per_m * (-move - cfg.backward_tol_m)
                continue
            if reach is None:
                reach = graph.distances_from(ta_node, limit)
            sp = reach.get(fr[b])
            if sp is None:
                continue  # unreachable within radius -> disconnected
            d_route[i, j] = (1.0 - ta) * la + sp + tb * ln[b]
            if fr[b] == ta_node and to[b] == fa:
                extra[i, j] += cfg.uturn_penalty
    finite = np.isfinite(d_route)
    logp = np.full((ka, kb), cfg.disconnected_logp)
    dr = d_route[finite]
    speed = dr / max(dt, 1e-3)
    logp[finite] = (
        -np.abs(dr - d_gps) / cfg.beta_m
        - cfg.speed_penalty_per_mps * np.maximum(0.0, speed - cfg.max_speed_mps)
        - extra[finite]
    )
    return logp
