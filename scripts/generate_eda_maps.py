"""
Generate interactive Folium maps for sample trips in train and student_test.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pathsnap_loader import load_trip
from src.visualize_folium import create_trip_map

def main():
    out_dir = PROJECT_ROOT / "outputs" / "figures" / "eda_maps"
    out_dir.mkdir(parents=True, exist_ok=True)

    trips_to_map = [
        ("00000001", "train"),
        ("00000002", "train"),
        ("00000000", "student_test"),
        ("00000052", "student_test"),
    ]

    for tid, partition in trips_to_map:
        print(f"Generating Folium map for {partition} trip {tid}...")
        trip_data = load_trip(tid, partition=partition)
        out_html = out_dir / f"trip_{tid}_{partition}_map.html"
        create_trip_map(trip_data, out_html=out_html)

    print("\nAll sample EDA Folium maps generated successfully!")

if __name__ == "__main__":
    main()
