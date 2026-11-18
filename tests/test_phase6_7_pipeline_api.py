"""Phase 6 pipeline + Phase 7 API integration tests on a real student-test trip.

python -m unittest tests.test_phase6_7_pipeline_api -v
"""

import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TRIP = "00000052"  # smallest student-test trip (262 points)
RAW_TRIP = ROOT / "data" / "raw" / "Student folder" / "student_test" / TRIP
MODEL = ROOT / "artifacts" / "models" / "lightgbm.joblib"
SHARD = ROOT / "data" / "processed" / "phase3" / "student_test" / "candidates" / f"{TRIP}.parquet"
HAVE_DATA = RAW_TRIP.exists() and MODEL.exists()


@unittest.skipUnless(HAVE_DATA, "raw student-test data or model missing")
class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.phase6_pipeline.end_to_end_matcher import PathSnapPipeline

        cls.pipe = PathSnapPipeline()
        cls.res = cls.pipe.run(TRIP, "student_test", evaluate=True, include_geometry=True)

    def test_model_bundle_matches_feature_contract(self):
        from src.phase3_features.feature_engineering import FEATURE_COLUMNS

        b = joblib.load(MODEL)
        self.assertEqual(list(b["features"]), list(FEATURE_COLUMNS))
        self.assertTrue(hasattr(b["model"], "predict_proba"))

    @unittest.skipUnless(SHARD.exists(), "processed Phase 3 shard missing")
    def test_ml_only_equals_saved_model_on_stored_phase3_features(self):
        from src.phase3_features.feature_engineering import FEATURE_COLUMNS
        from src.phase5_hmm.hmm_decoder import ml_only_argmax

        df = pd.read_parquet(SHARD)
        p = joblib.load(MODEL)["model"].predict_proba(df[FEATURE_COLUMNS].to_numpy(np.float32))[:, 1]
        ref = ml_only_argmax(pd.DataFrame({"point_idx": df.point_idx, "arc_id": df.arc_id, "prob": p}), self.res["n_points"])
        self.assertEqual(ref.tolist(), self.res["ml_only_arc_per_point"])

    def test_one_valid_prediction_per_point(self):
        r = self.res
        self.assertEqual(len(r["hmm_arc_per_point"]), r["n_points"])
        self.assertEqual(sum(a >= 0 for a in r["hmm_arc_per_point"]), r["n_points_with_candidates"])
        self.assertGreater(len(r["hmm_route_arcs"]), 0)
        self.assertGreater(len(r["hmm_geometry"]), 0)
        for line in r["hmm_geometry"]:
            for lat, lon in line:
                self.assertTrue(-90 <= lat <= 90 and -180 <= lon <= 180)

    def test_metrics_are_real_and_in_range(self):
        m = self.res["metrics"]
        for k in ("ml_only", "hmm"):
            self.assertEqual(m[k]["points"], m["labelled_points"])
            self.assertTrue(0.0 <= m[k]["top1_accuracy"] <= 1.0)
            self.assertTrue(0.0 <= m[k]["continuity_reach3"] <= 1.0)
        self.assertGreater(self.res["timing_ms"]["inference_total"], 0)

    def test_predictions_do_not_depend_on_evaluation_labels(self):
        r2 = self.pipe.run(TRIP, "student_test", evaluate=False, include_geometry=True)
        self.assertIsNone(r2["metrics"])
        self.assertEqual(r2["hmm_arc_per_point"], self.res["hmm_arc_per_point"])
        self.assertEqual(r2["ml_only_arc_per_point"], self.res["ml_only_arc_per_point"])

    def test_json_serializable(self):
        s = json.dumps(self.res)
        self.assertEqual(json.loads(s)["trip_id"], TRIP)

    def test_rejects_bad_trip_ids(self):
        with self.assertRaises(ValueError):
            self.pipe.run("../etc", "student_test")
        with self.assertRaises(FileNotFoundError):
            self.pipe.run("99999999", "student_test")


@unittest.skipUnless(HAVE_DATA, "raw student-test data or model missing")
class TestAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi.testclient import TestClient

        from api.main import app

        cls.ctx = TestClient(app)
        cls.client = cls.ctx.__enter__()  # runs the startup lifespan (model load)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_health(self):
        h = self.client.get("/health").json()
        self.assertEqual(h["status"], "ok")
        self.assertTrue(h["model_loaded"])
        self.assertEqual(h["trips_available"], 10)

    def test_trips(self):
        t = self.client.get("/trips").json()
        self.assertEqual(t["count"], 10)
        self.assertIn(TRIP, [x["trip_id"] for x in t["trips"]])

    def test_predict_real_trip(self):
        r = self.client.post("/predict", json={"trip_id": TRIP})
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertEqual(body["trip_id"], TRIP)
        self.assertEqual(len(body["gps"]), body["n_points"])
        self.assertTrue(body["hmm_geometry"] and body["reference_geometry"])
        self.assertIsNotNone(body["metrics"]["hmm"]["top1_accuracy"])
        again = self.client.post("/predict", json={"trip_id": TRIP}).json()
        self.assertTrue(again["cached"])
        self.assertEqual(again["hmm_arc_per_point"], body["hmm_arc_per_point"])

    def test_predict_errors(self):
        self.assertEqual(self.client.post("/predict", json={"trip_id": "99999999"}).status_code, 404)
        self.assertEqual(self.client.post("/predict", json={"trip_id": "../x"}).status_code, 422)
        self.assertEqual(self.client.post("/predict", json={}).status_code, 422)

    def test_metrics(self):
        m = self.client.get("/metrics").json()
        self.assertEqual(m["model_comparison"]["best_model"], "lightgbm")
        self.assertEqual(len(m["model_comparison"]["models"]), 5)

    def test_ui_served(self):
        r = self.client.get("/ui/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("Run Map Matching", r.text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
