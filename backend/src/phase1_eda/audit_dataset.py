"""
Run full dataset audit and generate exploratory report.
"""

from pathlib import Path
import sys
import json
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pathsnap_loader import load_trip, audit_dataset

def main():
    print("Testing single trip load (00000001)...")
    t1 = load_trip("00000001", partition="train")
    print(f"Loaded Trip {t1.trip_id}:")
    print(f"  Track rows: {len(t1.track_df)}")
    print(f"  Nodes: {len(t1.nodes_df)}")
    print(f"  Arcs: {len(t1.arcs_df)}")
    print(f"  Route: {len(t1.route)} arcs")

    print("\nAuditing all 80 train trips and 10 student-test trips...")
    audit = audit_dataset()

    out_file = Path("outputs/reports/dataset_exploration_report.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    print(f"\nSaved audit report to {out_file}")

    for p in ["train", "student_test"]:
        stats = audit["statistics"][p]
        pts = stats["track_points"]
        nds = stats["nodes"]
        arcs = stats["arcs"]
        rt = stats["route_length"]
        hz = stats["sampling_rate_1hz_pct"]
        print(f"\n=== Partition: {p} ({len(pts)} trips) ===")
        print(f"  GPS track points: min={min(pts)}, median={np.median(pts):.0f}, mean={np.mean(pts):.1f}, max={max(pts)}")
        print(f"  Road nodes: min={min(nds)}, median={np.median(nds):.0f}, mean={np.mean(nds):.1f}, max={max(nds)}")
        print(f"  Road arcs: min={min(arcs)}, median={np.median(arcs):.0f}, mean={np.mean(arcs):.1f}, max={max(arcs)}")
        print(f"  Route length: min={min(rt)}, median={np.median(rt):.0f}, mean={np.mean(rt):.1f}, max={max(rt)}")
        print(f"  1 Hz sampling %: mean={np.mean(hz):.2f}%")

    print(f"\nData quality issues detected: {len(audit['data_quality_issues'])}")
    for iss in audit["data_quality_issues"][:10]:
        print(f"  - {iss}")

if __name__ == "__main__":
    main()
