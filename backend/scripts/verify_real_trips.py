"""
Verify FastAPI endpoints and real end-to-end inference across Short, Medium, Long trips.
"""

from pathlib import Path
import sys
import time
import json
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.phase7_api.app import app

def main():
    with TestClient(app) as client:
        print("=" * 60)
        print("TASK 2: FINAL END-TO-END VERIFICATION")
        print("=" * 60)

        # 1. Health
        r = client.get("/health")
        assert r.status_code == 200, f"Health check failed: {r.status_code}"
        health_data = r.json()
        print(f"[PASS] GET /health -> status: {health_data['status']}, model: {health_data['model']}")
        assert health_data["status"] == "ok"
        assert health_data["model_loaded"] is True

        # 2. Trips
        r = client.get("/trips")
        assert r.status_code == 200
        trips_data = r.json()
        print(f"[PASS] GET /trips -> found {trips_data['count']} student-test trips")
        assert trips_data["count"] == 10

        # 3. Metrics
        r = client.get("/metrics")
        assert r.status_code == 200
        metrics_data = r.json()
        print(f"[PASS] GET /metrics -> model_comparison: {bool(metrics_data['model_comparison'])}, hmm_eval: {bool(metrics_data['hmm_evaluation'])}")

        # 4. Test 3 Real Trips: Short, Medium, Long
        test_trips = [
            ("00000052", "Short", 262),
            ("00000007", "Medium", 1354),
            ("00000040", "Long", 2995),
        ]

        results_summary = []

        for tid, category, expected_pts in test_trips:
            print(f"\n--- Testing Trip {tid} ({category}, ~{expected_pts} points) ---")
            t0 = time.perf_counter()
            r = client.post("/predict", json={"trip_id": tid, "include_geometry": True})
            elapsed = time.perf_counter() - t0
            assert r.status_code == 200, f"Predict failed for {tid}: {r.status_code} {r.text}"
            data = r.json()

            # Validations
            n_pts = data["n_points"]
            ml_acc = data["metrics"]["ml_only"]["top1_accuracy"]
            hmm_acc = data["metrics"]["hmm"]["top1_accuracy"]
            ml_cont = data["metrics"]["ml_only"]["continuity_reach3"]
            hmm_cont = data["metrics"]["hmm"]["continuity_reach3"]
            inf_ms = data["timing_ms"]["inference_total"]
            pts_changed = data["points_changed_by_hmm"]

            print(f"  Points: {n_pts}")
            print(f"  Inference Time: {inf_ms:.1f} ms (server roundtrip: {data['server_ms']} ms)")
            print(f"  Points Corrected by HMM: {pts_changed} ({pts_changed/n_pts*100:.1f}%)")
            print(f"  ML-only Top-1 Accuracy: {ml_acc*100:.2f}% | Continuity (reach-3): {ml_cont*100:.2f}%")
            print(f"  HMM Top-1 Accuracy:     {hmm_acc*100:.2f}% | Continuity (reach-3): {hmm_cont*100:.2f}%")
            print(f"  Accuracy Delta: +{(hmm_acc - ml_acc)*100:.2f}%")

            # Check geometry
            assert "gps" in data and len(data["gps"]) == n_pts
            assert "ml_only_geometry" in data and len(data["ml_only_geometry"]) > 0
            assert "hmm_geometry" in data and len(data["hmm_geometry"]) > 0
            assert "reference_geometry" in data and len(data["reference_geometry"]) > 0

            # Check arc ids
            ml_arcs = data["ml_only_arc_per_point"]
            hmm_arcs = data["hmm_arc_per_point"]
            assert len(ml_arcs) == n_pts
            assert len(hmm_arcs) == n_pts
            assert all(isinstance(a, int) for a in ml_arcs)
            assert all(isinstance(a, int) for a in hmm_arcs)

            # Check route arcs
            assert len(data["ml_only_route_arcs"]) > 0
            assert len(data["hmm_route_arcs"]) > 0

            results_summary.append({
                "trip_id": tid,
                "category": category,
                "points": n_pts,
                "inf_ms": round(inf_ms, 1),
                "server_ms": data["server_ms"],
                "pts_changed": pts_changed,
                "ml_acc": round(ml_acc * 100, 2),
                "hmm_acc": round(hmm_acc * 100, 2),
                "ml_cont": round(ml_cont * 100, 2),
                "hmm_cont": round(hmm_cont * 100, 2),
            })

        # 5. UI serving
        r = client.get("/ui/")
        assert r.status_code == 200
        assert "PathSnap" in r.text
        print(f"\n[PASS] GET /ui/ -> 200 OK ({len(r.text)} bytes served)")

        print("\n" + "=" * 60)
        print("ALL END-TO-END VERIFICATION CHECKS PASSED!")
        print("=" * 60)
        print(json.dumps(results_summary, indent=2))

if __name__ == "__main__":
    main()
