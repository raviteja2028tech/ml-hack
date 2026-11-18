"""Phase 5 HMM/Viterbi tests (synthetic, no data needed).  python -m unittest tests.test_phase5_hmm -v"""

import itertools
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.phase5_hmm.hmm_decoder import NO_ARC, HMMConfig, HMMRouteDecoder, ml_only_argmax, viterbi  # noqa: E402
from src.phase5_hmm.transition_model import RoadGraph, TransitionConfig, transition_logp  # noqa: E402


def brute_force(emis, trans):
    best, best_path = -np.inf, None
    for path in itertools.product(*[range(len(e)) for e in emis]):
        s = sum(emis[k][path[k]] for k in range(len(emis))) + sum(trans[k][path[k], path[k + 1]] for k in range(len(trans)))
        if s > best:
            best, best_path = s, path
    return np.array(best_path), best


class TestViterbi(unittest.TestCase):
    def test_sequence_optimum_differs_from_independent_argmax(self):
        emis = [np.log([0.6, 0.4]), np.log([0.45, 0.55]), np.log([0.6, 0.4])]
        stay = np.log(np.array([[0.9, 0.1], [0.1, 0.9]]))
        path, _ = viterbi(emis, [stay, stay])
        independent = [int(np.argmax(e)) for e in emis]
        self.assertEqual(independent, [0, 1, 0])
        self.assertEqual(path.tolist(), [0, 0, 0])  # switching costs more than the small emission gain

    def test_matches_brute_force_with_backtracking(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            ks = rng.integers(1, 4, size=rng.integers(1, 6))
            emis = [rng.normal(size=k) for k in ks]
            trans = [rng.normal(size=(ks[i], ks[i + 1])) for i in range(len(ks) - 1)]
            path, score = viterbi(emis, trans)
            bp, bs = brute_force(emis, trans)
            self.assertAlmostEqual(score, bs, places=9)
            self.assertEqual(path.tolist(), bp.tolist())

    def test_shape_validation_and_empty(self):
        self.assertEqual(len(viterbi([], [])[0]), 0)
        with self.assertRaises(ValueError):
            viterbi([np.zeros(2), np.zeros(3)], [np.zeros((3, 2))])
        with self.assertRaises(ValueError):
            viterbi([np.zeros(2), np.zeros(3)], [])


def toy_graph():
    # nodes along x: 0 -(arc0)-> 1 -(arc1)-> 2 ; arc2: isolated 3 -> 4 parallel to arc1; arc3: reverse of arc0
    fr = [0, 1, 3, 1]
    to = [1, 2, 4, 0]
    ln = [10.0, 10.0, 10.0, 10.0]
    return RoadGraph(fr, to, ln, n_nodes=5)


class TestTransitions(unittest.TestCase):
    def test_directed_shortest_paths(self):
        g = toy_graph()
        self.assertEqual(g.distances_from(0, 100.0)[2], 20.0)
        self.assertNotIn(3, g.distances_from(0, 100.0))  # isolated component unreachable
        self.assertNotIn(0, g.distances_from(2, 100.0))  # directed: no way back from node 2

    def test_consistent_move_scores_best_and_disconnected_is_penalised(self):
        g, cfg = toy_graph(), TransitionConfig()
        # from middle of arc0 to middle of arc1 = 5 + 5 = 10 m of road; GPS moved 10 m
        lp = transition_logp(g, [0], [0.5], [1, 2], [0.5, 0.5], d_gps=10.0, dt=1.0, cfg=cfg)
        self.assertAlmostEqual(lp[0, 0], 0.0, places=9)
        self.assertEqual(lp[0, 1], cfg.disconnected_logp)

    def test_speed_backward_and_uturn_penalties(self):
        g, cfg = toy_graph(), TransitionConfig()
        same_fwd = transition_logp(g, [0], [0.2], [0], [0.6], d_gps=4.0, dt=1.0, cfg=cfg)[0, 0]
        same_back = transition_logp(g, [0], [0.9], [0], [0.1], d_gps=8.0, dt=1.0, cfg=cfg)[0, 0]
        self.assertAlmostEqual(same_fwd, 0.0, places=9)
        self.assertLess(same_back, same_fwd)  # moving 8 m backwards along one arc is penalised
        fast = transition_logp(g, [0], [0.0], [1], [1.0], d_gps=20.0, dt=0.1, cfg=cfg)[0, 0]  # 200 m/s
        self.assertLess(fast, -50)
        uturn = transition_logp(g, [0], [1.0], [3], [0.0], d_gps=0.0, dt=1.0, cfg=cfg)[0, 0]
        self.assertAlmostEqual(uturn, -cfg.uturn_penalty, places=9)


class TestDecoder(unittest.TestCase):
    def setUp(self):
        self.g = toy_graph()
        # 4 GPS points moving along x at 1 Hz: on arc0 (x=2, 6) then arc1 (x=12, 16)
        self.px = np.array([2.0, 6.0, 12.0, 16.0])
        self.py = np.zeros(4)
        self.ts = np.arange(4, dtype=float)
        self.seg = np.zeros(4, dtype=int)

    def cands(self, rows):
        return pd.DataFrame(rows, columns=["point_idx", "arc_id", "proj_t", "prob"])

    def test_connectivity_overrides_slightly_higher_disconnected_emission(self):
        c = self.cands([
            (0, 0, 0.2, 0.9), (0, 2, 0.2, 0.1),
            (1, 0, 0.6, 0.9), (1, 2, 0.6, 0.1),
            (2, 1, 0.2, 0.45), (2, 2, 0.2, 0.55),  # ML-only would jump to the isolated arc 2
            (3, 1, 0.6, 0.9), (3, 2, 0.6, 0.1),
        ])
        ml = ml_only_argmax(c, 4)
        res = HMMRouteDecoder(self.g).decode(c, self.px, self.py, self.ts, self.seg)
        self.assertEqual(ml.tolist(), [0, 0, 2, 1])
        self.assertEqual(res.arc_id.tolist(), [0, 0, 1, 1])
        self.assertEqual(res.n_disconnected_steps, 0)

    def test_missing_candidates_duplicates_and_chain_breaks(self):
        c = self.cands([
            (0, 0, 0.2, 0.8), (0, 0, 0.2, 0.3),  # duplicate (point, arc) row
            (2, 1, 0.2, 0.7), (2, 2, 0.2, 0.3),  # point 1 has no candidates at all
            (3, 1, 0.6, 0.0),                   # probability 0 -> clipped, still finite
        ])
        seg = np.array([0, 0, 0, 1])  # point 3 is in a new time segment
        res = HMMRouteDecoder(self.g).decode(c, self.px, self.py, self.ts, seg)
        self.assertEqual(res.arc_id[1], NO_ARC)
        self.assertEqual(res.n_no_candidates, 1)
        self.assertEqual(res.n_chains, 2)
        self.assertEqual(res.arc_id.tolist(), [0, NO_ARC, 1, 1])
        self.assertTrue(np.isfinite(res.log_score))

    def test_long_gap_restarts_chain_and_predictions_are_candidates(self):
        c = self.cands([(i, a, 0.5, p) for i in range(4) for a, p in ((0, 0.6), (2, 0.4))])
        ts = np.array([0.0, 1.0, 100.0, 101.0])  # 99 s gap
        res = HMMRouteDecoder(self.g, HMMConfig(max_gap_s=30.0)).decode(c, self.px, self.py, ts, self.seg)
        self.assertEqual(res.n_chains, 2)
        keys = set(zip(c.point_idx, c.arc_id))
        self.assertTrue(all((i, int(a)) in keys for i, a in enumerate(res.arc_id)))

    def test_decoder_never_needs_labels(self):
        c = self.cands([(0, 0, 0.2, 0.9), (1, 0, 0.6, 0.9)])
        c["label"] = 1  # extra columns are ignored
        res = HMMRouteDecoder(self.g).decode(c, self.px[:2], self.py[:2], self.ts[:2], self.seg[:2])
        self.assertEqual(res.arc_id.tolist(), [0, 0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
