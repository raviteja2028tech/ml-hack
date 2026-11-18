"""
Phase 5: Hidden Markov Model + Viterbi route reconstruction.

Hidden states     candidate road arcs of each GPS point (from the Phase 3 candidate generator)
Emissions         log of the saved classifier's probability for that candidate (clipped, stable)
Transitions       network distance vs GPS displacement, speed, direction (see transition_model.py)
Decoding          exact Viterbi dynamic programming in log space, with back-pointers

Inference never reads the reference route or labels: inputs are candidate ids, projection
positions, classifier probabilities and the GPS points themselves.

Edge cases:
* points with no candidate arc  -> arc_id -1, status "no_candidates" (nothing is invented)
* duplicate (point, arc) rows   -> deduplicated, highest probability kept
* time gaps > max_gap_s or a change of time segment -> the chain restarts (independent Viterbi)
* unreachable transitions       -> finite ``disconnected_logp`` so decoding never fails
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.phase5_hmm.transition_model import RoadGraph, TransitionConfig, transition_logp

NO_ARC = -1


def viterbi(emissions: Sequence[np.ndarray], transitions: Sequence[np.ndarray]) -> Tuple[np.ndarray, float]:
    """
    Exact max-sum Viterbi.

    emissions[k]   : (K_k,) log-emission scores of step k
    transitions[k] : (K_k, K_{k+1}) log-transition scores from step k to k+1 (len = n_steps - 1)
    returns        : (best state index per step, best total log-score)
    """
    n = len(emissions)
    if n == 0:
        return np.empty(0, dtype=np.int64), 0.0
    if len(transitions) != n - 1:
        raise ValueError(f"need {n - 1} transition matrices, got {len(transitions)}")
    score = np.asarray(emissions[0], dtype=np.float64).copy()
    back: List[np.ndarray] = []
    for k in range(1, n):
        trans = np.asarray(transitions[k - 1], dtype=np.float64)
        if trans.shape != (len(score), len(emissions[k])):
            raise ValueError(f"transition {k - 1} has shape {trans.shape}, expected {(len(score), len(emissions[k]))}")
        total = score[:, None] + trans
        bp = np.argmax(total, axis=0)
        score = total[bp, np.arange(total.shape[1])] + np.asarray(emissions[k], dtype=np.float64)
        back.append(bp)
    path = np.empty(n, dtype=np.int64)
    path[-1] = int(np.argmax(score))
    best = float(score[path[-1]])
    for k in range(n - 1, 0, -1):
        path[k - 1] = back[k - 1][path[k]]
    return path, best


@dataclass(frozen=True)
class HMMConfig:
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    emission_eps: float = 1e-6
    emission_weight: float = 1.0
    max_gap_s: float = 30.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class DecodeResult:
    arc_id: np.ndarray  # (n_points,) decoded arc per GPS point, NO_ARC if no candidates
    prob: np.ndarray  # classifier probability of the decoded candidate (nan if none)
    chain_id: np.ndarray  # independent Viterbi chain of each point (-1 if none)
    n_chains: int
    n_decoded: int
    n_no_candidates: int
    n_disconnected_steps: int  # decoded transitions that had to cross an unreachable gap
    log_score: float


class HMMRouteDecoder:
    def __init__(self, graph: RoadGraph, config: HMMConfig = HMMConfig()):
        self.graph = graph
        self.config = config

    def decode(
        self,
        candidates: pd.DataFrame,
        px: np.ndarray,
        py: np.ndarray,
        ts: np.ndarray,
        segment_id: np.ndarray,
    ) -> DecodeResult:
        """
        candidates: rows (point_idx, arc_id, proj_t, prob) for any subset of the GPS points.
        px, py, ts, segment_id: per-GPS-point metric coordinates, timestamps, time segments.
        """
        cfg = self.config
        n_points = len(px)
        out_arc = np.full(n_points, NO_ARC, dtype=np.int64)
        out_prob = np.full(n_points, np.nan)
        out_chain = np.full(n_points, -1, dtype=np.int64)

        c = candidates[["point_idx", "arc_id", "proj_t", "prob"]]
        c = c.sort_values(["point_idx", "prob"], ascending=[True, False], kind="stable")
        c = c.drop_duplicates(["point_idx", "arc_id"], keep="first")
        pidx = c["point_idx"].to_numpy(np.int64)
        if len(pidx) and (pidx.min() < 0 or pidx.max() >= n_points):
            raise ValueError("candidate point_idx outside the track")
        arcs = c["arc_id"].to_numpy(np.int64)
        tpos = c["proj_t"].to_numpy(np.float64)
        logp = cfg.emission_weight * np.log(np.clip(c["prob"].to_numpy(np.float64), cfg.emission_eps, 1.0))
        probs = c["prob"].to_numpy(np.float64)

        obs, start = np.unique(pidx, return_index=True)
        end = np.r_[start[1:], len(pidx)]

        # split observed points into chains
        chains: List[List[int]] = []
        for k, p in enumerate(obs):
            if k == 0:
                chains.append([k])
                continue
            q = obs[k - 1]
            if segment_id[p] != segment_id[q] or (ts[p] - ts[q]) > cfg.max_gap_s:
                chains.append([k])
            else:
                chains[-1].append(k)

        total_score = 0.0
        n_disc = 0
        for cid, chain in enumerate(chains):
            emis, trans = [], []
            for n_k, k in enumerate(chain):
                s, e = start[k], end[k]
                emis.append(logp[s:e])
                if n_k:
                    k0 = chain[n_k - 1]
                    s0, e0 = start[k0], end[k0]
                    p0, p1 = obs[k0], obs[k]
                    d_gps = math.hypot(px[p1] - px[p0], py[p1] - py[p0])
                    dt = float(ts[p1] - ts[p0])
                    trans.append(
                        transition_logp(self.graph, arcs[s0:e0], tpos[s0:e0], arcs[s:e], tpos[s:e], d_gps, dt, cfg.transition)
                    )
            path, score = viterbi(emis, trans)
            total_score += score
            for n_k, k in enumerate(chain):
                row = start[k] + path[n_k]
                p = obs[k]
                out_arc[p] = arcs[row]
                out_prob[p] = probs[row]
                out_chain[p] = cid
                if n_k and trans[n_k - 1][path[n_k - 1], path[n_k]] <= cfg.transition.disconnected_logp:
                    n_disc += 1

        return DecodeResult(
            arc_id=out_arc,
            prob=out_prob,
            chain_id=out_chain,
            n_chains=len(chains),
            n_decoded=int(len(obs)),
            n_no_candidates=int(n_points - len(obs)),
            n_disconnected_steps=int(n_disc),
            log_score=float(total_score),
        )


def ml_only_argmax(candidates: pd.DataFrame, n_points: int) -> np.ndarray:
    """Independent per-point argmax of the classifier probability (the ML-only baseline)."""
    out = np.full(n_points, NO_ARC, dtype=np.int64)
    if len(candidates) == 0:
        return out
    c = candidates.sort_values(["point_idx", "prob"], ascending=[True, False], kind="stable")
    first = c.drop_duplicates("point_idx", keep="first")
    out[first["point_idx"].to_numpy(np.int64)] = first["arc_id"].to_numpy(np.int64)
    return out
