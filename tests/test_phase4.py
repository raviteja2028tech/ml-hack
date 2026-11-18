"""Phase 4 tests: metrics, split integrity, leakage guards.  python -m unittest tests.test_phase4 -v"""

import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train_models import MODEL_ORDER, ArcTopology, make_model, route_continuity, top1_rows, trip_balanced_subset  # noqa: E402

RESULTS = ROOT / "outputs" / "reports" / "phase4"


class TestMetrics(unittest.TestCase):
    def test_top1_picks_argmax_per_point(self):
        df = pd.DataFrame({"trip_id": ["a", "a", "a", "b", "b"], "point_idx": [0, 0, 1, 0, 0], "label": [0, 1, 1, 1, 0]})
        p = np.array([0.2, 0.9, 0.1, 0.3, 0.7])
        rows = sorted(top1_rows(df, p).tolist())
        self.assertEqual(rows, [1, 2, 4])
        self.assertAlmostEqual(df.iloc[rows].label.mean(), 2 / 3)

    def test_top1_same_point_idx_in_different_trips_not_merged(self):
        df = pd.DataFrame({"trip_id": ["a", "b"], "point_idx": [5, 5], "label": [1, 1]})
        self.assertEqual(len(top1_rows(df, np.array([0.1, 0.2]))), 2)

    def test_route_continuity_on_real_topology(self):
        topo = ArcTopology()
        fr, to, _, _ = topo.get("00000052")
        a0 = 0
        succ = int(np.flatnonzero(fr == to[a0])[0])
        unrelated = int(np.flatnonzero((fr != to[a0]) & (np.arange(len(fr)) != a0))[-1])
        pred = pd.DataFrame({"trip_id": "00000052", "point_idx": [0, 1, 2, 3], "segment_id": [0, 0, 0, 1],
                             "arc_id": [a0, a0, succ, unrelated]})
        c = route_continuity(pred, topo)
        self.assertEqual(c["pairs"], 2)  # pair (2,3) crosses a segment boundary -> excluded
        self.assertEqual(c["continuity_direct"], 1.0)

    def test_trip_balanced_subset_caps_points(self):
        df = pd.DataFrame({"trip_id": ["a"] * 30 + ["b"] * 4, "point_idx": list(np.repeat(np.arange(10), 3)) + [0, 0, 1, 1]})
        sub = trip_balanced_subset(df, cap=4)
        n = sub.drop_duplicates(["trip_id", "point_idx"]).groupby("trip_id").size()
        self.assertEqual(n["a"], 4)
        self.assertEqual(n["b"], 2)
        self.assertEqual(len(sub[sub.trip_id == "a"]), 12)  # all candidates of a kept point retained

    def test_scaler_is_inside_pipeline(self):
        for name in ("logistic_regression", "mlp"):
            self.assertTrue(hasattr(make_model(name), "steps"), f"{name} preprocessing must be fitted per fold")


@unittest.skipUnless((RESULTS / "model_comparison.json").exists(), "phase 4 not run yet")
class TestResults(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cmp = json.loads((RESULTS / "model_comparison.json").read_text(encoding="utf-8"))
        cls.res = {m: json.loads((RESULTS / f"{m}.json").read_text(encoding="utf-8")) for m in MODEL_ORDER}

    def test_five_models_from_distinct_families(self):
        self.assertEqual(len(self.cmp["comparison"]), 5)
        self.assertEqual(len({r["family"] for r in self.cmp["comparison"]}), 5)

    def test_cv_folds_disjoint_and_cover_80_trips(self):
        for m, r in self.res.items():
            folds = r["cv"]["folds"]
            self.assertEqual(len(folds), 5, m)
            self.assertEqual(sum(f["val_trips"] for f in folds), 80, m)

    def test_metrics_valid_and_artifacts_saved(self):
        for m, r in self.res.items():
            t = r["student_test"]
            for k in ("top1_accuracy", "precision", "recall", "f1", "roc_auc", "pred_continuity_reach3"):
                self.assertTrue(0.0 <= t[k] <= 1.0, (m, k))
            self.assertTrue(np.isfinite(t["log_loss"]))
            self.assertEqual(t["points"], 19133)  # all labelled student-test points evaluated
            self.assertTrue((ROOT / r["model_file"]).exists(), m)

    def test_best_model_selected_by_cv_not_test(self):
        best = max(self.cmp["comparison"], key=lambda r: (r["cv_top1_mean"], -r["cv_logloss_mean"]))
        self.assertEqual(best["model"], self.cmp["best_model"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
