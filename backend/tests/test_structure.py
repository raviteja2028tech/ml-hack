"""
Verification tests for the reorganized PathSnap phase-wise structure.
"""

from pathlib import Path
import sys
import unittest
import joblib

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class TestReorganizedStructure(unittest.TestCase):

    def test_directory_structure_exists(self):
        expected_dirs = [
            "data/raw",
            "data/processed",
            "src/phase1_eda",
            "src/phase2_preprocessing",
            "src/phase3_features",
            "src/phase4_models",
            "src/phase5_hmm",
            "src/phase6_pipeline",
            "api",
            "ui",
            "artifacts/models",
            "results/figures",
            "results/metrics",
            "results/maps",
            "tests",
            "docs",
            "reports",
        ]
        for d in expected_dirs:
            p = ROOT / d
            self.assertTrue(p.exists(), f"Directory missing: {d}")

    def test_models_exist_and_load(self):
        model_names = [
            "extra_trees.joblib",
            "lightgbm.joblib",
            "logistic_regression.joblib",
            "mlp.joblib",
            "random_forest.joblib",
        ]
        for m in model_names:
            mp = ROOT / "artifacts" / "models" / m
            self.assertTrue(mp.exists(), f"Model artifact missing: {m}")
            model = joblib.load(mp)
            self.assertIsNotNone(model, f"Failed to load model: {m}")

    def test_phase1_eda_imports(self):
        from src.phase1_eda.pathsnap_loader import load_trip, TripData
        t = load_trip("00000001", partition="train")
        self.assertIsInstance(t, TripData)
        self.assertEqual(len(t.track_df), 1315)

    def test_phase3_features_imports(self):
        from src.phase3_features.candidate_generator import generate_candidates, CandidateConfig
        from src.phase3_features.feature_engineering import build_candidate_features, FEATURE_COLUMNS
        self.assertIsNotNone(generate_candidates)
        self.assertIsNotNone(CandidateConfig)
        self.assertIsNotNone(build_candidate_features)
        self.assertGreater(len(FEATURE_COLUMNS), 0)

    def test_phase4_models_imports(self):
        from src.phase4_models.train_models import MODEL_ORDER, make_model
        self.assertEqual(len(MODEL_ORDER), 5)

    def test_api_service_imports(self):
        from api.main import app
        self.assertEqual(app.title, "PathSnap Map-Matching API")

    def test_results_and_reports_exist(self):
        self.assertTrue((ROOT / "results/maps/REVIEW1_MAP.html").exists())
        self.assertTrue((ROOT / "results/metrics/model_comparison.json").exists())
        self.assertTrue((ROOT / "reports/PHASE3_REPORT.md").exists())
        self.assertTrue((ROOT / "reports/DATASET_EXPLORATION.md").exists())


if __name__ == "__main__":
    unittest.main()
