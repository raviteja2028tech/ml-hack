# PATHSNAP: DATASET EXPLORATION & AUDIT REPORT

**Date**: 2026-11-18  
**Partitions Audited**: Training (80 trips) & Student-Testing (10 trips)  
**Total Trips Audited**: 90 trips  
**Audit Output**: [`outputs/reports/dataset_exploration_report.json`](file:///c:/Users/ravit/OneDrive/Desktop/ml-hackthon/outputs/reports/dataset_exploration_report.json)  

---

> **Phase 3 erratum (2026-11-18)** — measured over all trips in `docs/PHASE3_REPORT.md` §8:
> - Route connectivity is 100 % in **79/90** trips, not 89/90. Trips 02, 19, 33, 41, 42, 69, 75, 80, 86, 89 (train) and 76 (student-test) have at least one break (98.13–99.93 %). The audit below only flagged trips under 99 %.
> - The "< 0.5 % candidate miss rate" was trip 1 only. Over 80 train trips at R = 50 m it is 1.79 % pooled (excluding route-end points), with a 0.00 % per-trip median; trips 54/33/02 dominate.
> - Train GPS points total **208,417** (the JSON report is correct; the table text below said 208,416).

## 1. Executive Summary & Inventory

Each trip directory is completely self-contained and consists of four raw, headerless, whitespace-separated files:
1. `<trip_id>.track`: GPS observations (longitude, latitude, timestamp).
2. `<trip_id>.nodes`: Road network intersection nodes (longitude, latitude) with 0-based row indexing defining `node_id`.
3. `<trip_id>.arcs`: Directed road network segments (`from_node_id`, `to_node_id`) with 0-based row indexing defining `arc_id`.
4. `<trip_id>.route`: Ground-truth chronological sequence of traversed `arc_id` values.

### Summary Statistics Table

| Partition | Metric | Min | Median | Mean | Max |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Training (80 Trips)** | **GPS Track Points** | 361 | 1,529 | 2,605.2 | 19,313 |
| | **Road Network Nodes** | 299 | 9,608 | 24,168.6 | 155,003 |
| | **Road Network Arcs** | 649 | 19,526 | 48,461.3 | 307,392 |
| | **Ground Truth Route Length** | 59 | 280 | 391.7 | 2,266 |
| | **1 Hz Sampling Ratio** | 82.4% | 96.1% | 93.84% | 99.8% |
| **Student Testing (10 Trips)** | **GPS Track Points** | 262 | 1,420 | 1,916.0 | 3,958 |
| | **Road Network Nodes** | 1,770 | 6,596 | 9,068.9 | 26,635 |
| | **Road Network Arcs** | 3,577 | 13,145 | 18,504.1 | 56,027 |
| | **Ground Truth Route Length** | 95 | 250 | 347.7 | 1,323 |
| | **1 Hz Sampling Ratio** | 94.3% | 98.7% | 98.20% | 99.7% |

---

## 2. Key Geospatial & Topological Discoveries

1. **Trip-Specific Road Networks**:
   - Each trip includes a cropped subgraph of the regional road network surrounding the bounding box of that trajectory.
   - `arc_id` and `node_id` indices are trip-local. All models must evaluate candidate segments based on geospatial and topological features relative to the trip's local network rather than treating `arc_id` as a global categorical label.

2. **100% Route Topological Continuity**:
   - Across 89 of the 90 audited trips, the ground-truth route exhibits **100.0% topological connectivity**: for every step $k$, `to_node(route[k]) == from_node(route[k+1])`.
   - Trip `00000075` is the sole exception with 98.1% connectivity due to a minor 2-segment gap in the reference data.

3. **GPS Distance to True Road Segments**:
   - For sample trip `00000001`, the median perpendicular distance from raw GPS points to the true road segment is **4.29 meters** (mean: 5.00 meters, 95th percentile: 14.8 meters, max: 27.15 meters).
   - This validates that a spatial candidate radius of **30–50 meters** captures the true road segment with a **Candidate Miss Rate of < 0.5%**.

4. **Temporal Alignment**:
   - One GPS point does *not* equal one route arc. An average vehicle spends 2.5 to 5.0 GPS seconds traversing an arc segment.
   - In urban stop-and-go traffic (e.g. at red lights), a vehicle may generate 20–60 GPS points while stationary on a single arc.
   - In high-speed highway segments, short transitional arcs (<5 meters) may be traversed between consecutive 1 Hz GPS readings.

---

## 3. Interactive Folium Visualizations

Multi-layer interactive Folium HTML maps have been generated in [`outputs/figures/eda_maps/`](file:///c:/Users/ravit/OneDrive/Desktop/ml-hackthon/outputs/figures/eda_maps/):
- **Training Trip 00000001**: [`trip_00000001_train_map.html`](file:///c:/Users/ravit/OneDrive/Desktop/ml-hackthon/outputs/figures/eda_maps/trip_00000001_train_map.html)
- **Training Trip 00000002**: [`trip_00000002_train_map.html`](file:///c:/Users/ravit/OneDrive/Desktop/ml-hackthon/outputs/figures/eda_maps/trip_00000002_train_map.html)
- **Student-Test Trip 00000000**: [`trip_00000000_student_test_map.html`](file:///c:/Users/ravit/OneDrive/Desktop/ml-hackthon/outputs/figures/eda_maps/trip_00000000_student_test_map.html)
- **Student-Test Trip 00000052**: [`trip_00000052_student_test_map.html`](file:///c:/Users/ravit/OneDrive/Desktop/ml-hackthon/outputs/figures/eda_maps/trip_00000052_student_test_map.html)

Each map includes:
- Blue markers for raw GPS observations (with interactive timestamp and velocity tooltips).
- Green line for the ground-truth reference route.
- Gray lines for the surrounding road network topology.
- Interactive LayerControl toggle.
