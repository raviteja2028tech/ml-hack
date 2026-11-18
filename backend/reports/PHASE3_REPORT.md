# PATHSNAP — PHASE 3 REPORT
## Candidate Generation, Feature Engineering & Training-Dataset Preparation

**Date**: 2026-11-18 | **Status**: COMPLETE (no models trained; Phase 4 not started)
**Machine**: Ryzen 5 7535HS, 12 logical CPUs, 16.5 GB RAM (3.1–4.2 GB free during runs), Python 3.12.10, CPU only, offline

Aggregate numbers come from the generated reports below. The trip-level off-route investigation (section 4) was done on `train_points.parquet`:
[`outputs/reports/phase3_dataset_report.json`](../outputs/reports/phase3_dataset_report.json) (per-trip and aggregate),
[`outputs/reports/phase3_benchmark.json`](../outputs/reports/phase3_benchmark.json),
[`outputs/reports/phase3_unit_tests.txt`](../outputs/reports/phase3_unit_tests.txt),
[`outputs/reports/phase3_build_log.txt`](../outputs/reports/phase3_build_log.txt).

---

## 1. Files

| File | Role |
| :--- | :--- |
| `src/candidate_generator.py` | Exact radius search for candidate arcs (densified-arc cKDTree, dual-tree query, exact segment distance) |
| `src/feature_engineering.py` | GPS kinematics, arc graph, 42 candidate features. **Never reads the route.** |
| `src/route_labeler.py` | Monotone DP alignment of GPS points to `.route` positions, label status, label-quality diagnostics |
| `src/phase3_pipeline.py` | Per-trip pipeline `process_trip()` (load → candidates → features → labels) |
| `src/phase3_dataset.py` | Read helper `load_candidates()` / `load_points()` for Phase 4+ (labelled-only by default) |
| `scripts/build_phase3_dataset.py` | Parallel build of all trips → Parquet shards + consolidated files + JSON report |
| `scripts/benchmark_phase3.py` | Runtime / memory / exactness / projection-accuracy benchmark |
| `tests/test_phase3.py` | 29 unit + integration tests (stdlib `unittest`; pytest is not installed offline) |
| `data/processed/phase3/` | Generated datasets (below) |

Reused unchanged: `src/pathsnap_loader.py` (`load_trip`, local metric projection, scalar reference geometry).

### Reproduce
```powershell
.venv\Scripts\python.exe scripts\build_phase3_dataset.py --workers 3     # ~110 s, all 90 trips
.venv\Scripts\python.exe -m unittest tests.test_phase3 -v               # ~7 s
.venv\Scripts\python.exe scripts\benchmark_phase3.py                    # ~5 min (brute force is slow on purpose)
```

---

## 2. Candidate generation

**Problem with a midpoint index.** Arcs are straight node-to-node segments, up to 1,709 m long. A point can be 5 m from an arc whose midpoint is 800 m away. Phase 2's `arc_kdtree` indexes midpoints. On full trips it recovers only **47.7–63.4 %** of the arcs truly within the radius (benchmark B), so it is not used.

**Method (exact by construction):**
1. Bounding-box crop of arcs to the track ± query radius (lossless).
2. Each arc is split into `ceil(L / 10 m)` sub-intervals and their centres are indexed. Every point on an arc is ≤ 5 m from a sample.
3. Lossless corridor filter: keep only samples in 250 m grid cells occupied by (or adjacent to) the track. On trip 13 this cuts 1.24 M samples to 81 k.
4. Dual-tree `cKDTree.sparse_distance_matrix` with radius R + 5 m. By the triangle inequality this cannot miss an arc within R.
5. Unique (point, arc) pairs, exact clamped point-to-segment distance, keep ≤ R. Rows are sorted by (point, distance).

`CandidateConfig(radius_m=50, sample_spacing_m=10, max_candidates=None)` is validated, and any 30–50 m radius works. **Arc IDs are the raw trip-local 0-based row numbers of `.arcs`**; a test checks this against the raw file.

**Exactness and speed** (benchmark B, full trips, compared with O(points × arcs) brute force):

| Trip | Points | Arcs | R | True pairs | Ours | Brute force | Speed-up | Missing / extra |
| :-- | --: | --: | --: | --: | --: | --: | --: | :--: |
| 00000001 | 1,315 | 165,790 | 50 m | 15,991 | 0.155 s | 13.8 s | 89× | 0 / 0 |
| 00000013 | 2,580 | 307,392 | 50 m | 24,575 | 0.287 s | 51.1 s | 178× | 0 / 0 |

**Distance accuracy.** The local equirectangular projection was compared with haversine on 40,000 sampled candidate pairs from 20 trips: median relative error 0.006 %, max 0.46 %, **max absolute error 0.21 m** (p99 0.08 m). Benchmark C has the details.

**Radius choice.** The dataset is built at **R = 50 m**. Smaller radii are exact subsets (filter `dist_m <= r`). Train, 80 trips:

| R | Candidates / point | Miss rate, all points | Miss rate, excl. route-end points | Per-trip median miss | Trips with 0 misses |
| --: | --: | --: | --: | --: | --: |
| 30 m | 5.27 | 12.51 % | 3.28 % | 0.13 % | 26 / 80 |
| 40 m | 7.02 | 11.66 % | 2.34 % | 0.00 % | 49 / 80 |
| **50 m** | **8.97** | **11.16 %** | **1.79 %** | **0.00 %** | **59 / 80** |

Student-test at 50 m: 10.94 candidates/point, miss rate **0.14 %** (r40 0.22 %, r30 0.78 %).
The pooled train miss rate is dominated by three trips where the GPS leaves the reference route (section 4); it is not a generator weakness. 50 m keeps the most recall for 1.7× the rows of 30 m.

---

## 3. Features (42 columns, `FEATURE_COLUMNS`)

All features are computed from `.track` + `.nodes/.arcs` only. Kinematics never cross a time gap of more than 30 s, which defines a segment.

| Group | Features |
| :-- | :-- |
| GPS-to-road distance | `dist_m`, `dist_rank`, `dist_gap_to_min_m`, `dist_ratio_to_min`, `n_candidates`, `min_dist_point_m` |
| Segment | `arc_length_m`, `proj_t`, `proj_beyond_end_m` |
| Heading difference | `heading_diff_deg` (directed 0–180), `heading_cos`, `heading_diff_undirected_deg`, `heading_valid` |
| Vehicle speed | `speed_mps` (centred), `speed_smooth_mps` (rolling median 5), `accel_mps2`, `along_speed_mps` (signed, along the arc), `cross_speed_mps`, `turn_rate_deg_s`, `dt_prev_s`, `dt_next_s`, `step_prev_m`, `step_next_m` |
| Road connectivity | `from_in_degree`, `to_out_degree`, `is_two_way`, `has_multi_edge`, `is_self_loop`, `in_prev_cands`, `in_next_cands`, `pred_in_prev_cands`, `succ_in_next_cands`, `hops_from_prev_anchor`, `hops_to_next_anchor`, `is_prev_anchor`, `is_next_anchor` |
| Temporal consistency | `persistence_ratio` (± 5 points), `win_mean_dist_m`, `win_max_dist_m`, `win_heading_cos_mean` (± 2 point trajectory fit), `has_prev_point`, `has_next_point` |

Design notes:
- **Heading** is the centred displacement i−1 → i+1. If that displacement is under 3 m (stopped), the last measured heading is carried forward and `heading_valid = 0`. Heading is what separates the two directions of a two-way road: on average 92 % of a trip's arcs are two-way, so distance ties are the norm.
- **Anchor**: a label-free per-point "best guess", argmin of `dist + 7.5·(1 − heading_cos)`. Hop features measure arc-graph hops (BFS depth ≤ 3; value 4 means farther or unknown) from the previous point's anchor and to the next point's anchor.
- NaN never reaches the table. Undefined values at segment edges get fixed sentinels (−1, or 90° for "no heading"), and `has_prev_point` / `has_next_point` / `heading_valid` flags record when that happened. Nothing is fitted, and dtypes are fixed (float32 / int8 / int16) so every shard has the same schema.

---

## 4. Supervised labels from `.route`

**Why not one-to-one.** On train, 185,161 labelled points map onto 31,336 route entries. Stopped vehicles emit tens of points per arc. Short arcs (unvisited route arcs have a trip-median length of 9.3 m) are crossed between samples. Routes revisit arcs (67 revisits across the train routes) and make U-turns.

**Method.** A monotone subsequence-DTW alignment assigns each GPS point a non-decreasing route position. It minimises `min(dist, 200 m) + 15 m · (1 − cos Δheading)/2`, where the heading term counts only when the vehicle is moving. Start and end are free. Cost is O(N·K) with prefix minima, and the largest case (trip 92: N = 7,892, K = 2,266) takes 2.1 s per alignment.

**Per-point status:**
- `ok`: aligned arc is among the candidates.
- `boundary`: an adjacent route arc (± 2 positions) costs within 1 m of the assigned one. The point is at a junction; the label is kept and flagged.
- `off_route`: aligned arc is **not** a candidate. These points get no positive label. Do not train on them.

Separately, `label_equiv = 1` marks a multi-edge twin: another arc ID with identical from/to nodes, which is geometrically indistinguishable.

### Labelling quality

| Metric | Train (80) | Student-test (10) |
| :-- | --: | --: |
| GPS points | 208,417 | 19,160 |
| Labelled (`ok` + `boundary`) | **185,161 (88.84 %)** | **19,133 (99.86 %)** |
| — of which `boundary` | 22,292 (12.0 % of labelled) | 2,558 (13.4 %) |
| `off_route` | 23,256 | 27 |
| — at the route's first/last arc (track outside route span) | 16,833 | 0 |
| — mid-route | 6,423 | 27 |
| Positives per labelled point | exactly 1 (max 1) | exactly 1 |
| Multi-edge twin rows (`label_equiv`) | 21 | 1 |
| Points with an indistinguishable negative at that instant (Δdist < 0.5 m, Δcos < 0.05) | 12,251 (6.6 %) | 1,303 (6.8 %) |
| GPS → aligned arc distance, median / p95 (trip medians) | 4.64 m / 14.21 m | 4.74 m / 15.78 m |
| Route coverage inside the matched span (mean) | 91.3 % | 91.1 % |
| Route-progress vs GPS-step error, median (trip median) | 1.00 m | 0.90 m |
| Steps with progress error > 25 m | 0.70 % | 0.52 % |
| Label agreement when the heading weight is doubled (15 → 30 m) | 99.08 % | 99.06 % |
| Label agreement with no heading term (15 → 0) | 97.45 % (worst trip 75: 83.1 %) | 96.74 % |

The label is insensitive to the exact heading weight (99 % agreement when doubled), but the heading term itself is essential on two-way roads: trip 75 is 100 % two-way and 17 % of its labels flip without it.

**Where the off-route points come from** (all inspected, none forced into a label):
- **Trip 54**: after t ≈ 2,875 s the route has ended, but the device logs for another ~4.6 h. It sits in a ~100 × 140 m area 70–200 m from any road, which looks like a parked vehicle with GPS jitter (~7 m/s spurious "speed"). 16,584 points. Also one ~900 s off-network excursion of 690 points.
- **Trip 33**: 3,042 points (55 min) on *mapped* roads (median 4 candidates) at the exact position where the reference route has a connectivity break. The `.route` omits that part of the drive.
- **Trip 02**: same pattern: 311 points at the route break after position 53, plus a 552-point tail after the route ends.
- 385 off-route points are isolated (runs of fewer than 5 points), which is consistent with GPS glitches.

In trips 33 and 02 the true road is probably among the candidates but carries label 0. That is why `load_candidates()` keeps only labelled points by default.

**Label-free heuristic baselines** (Top-1 on labelled points; the bar Phase 4 has to clear):
nearest-distance **55.6 %** train / 53.7 % student-test (two-way ties make it a coin flip), and distance + heading anchor **82.5 %** train / 85.8 % student-test.

### Leakage controls
1. Feature code never reads `trip.route` or imports `route_labeler`. A test checks the source AST.
2. `process_trip` sets `trip.route = None` before any feature or candidate computation.
3. **A test recomputes features with the route removed and asserts the frames are bit-identical.**
4. Candidates are purely spatial, so the candidate set does not depend on the label.
5. Neighbour features use neighbouring points' *candidates and GPS*, never their labels. This is the batch (whole-trip) setting the API will also run in.
6. No cross-trip statistics are fitted in Phase 3. Scalers and imputers belong to Phase 4 training folds.
7. `trip_id` is stored for GroupKFold. `arc_id` is an identifier only (trip-local) and is **not** a feature.
8. Train and student-test are built into separate files; a test asserts the 80 / 10 trip sets are disjoint.

---

## 5. Generated datasets (`data/processed/phase3/`)

| File | Rows | Size (zstd) | Content |
| :-- | --: | --: | :-- |
| `train_candidates.parquet` | 1,869,503 | 116.5 MB | ids (`trip_id, point_idx, timestamp, segment_id, arc_id`) + 42 features + `label, label_equiv, label_status` |
| `train_points.parquet` | 208,417 | 8.2 MB | lon/lat/time, `n_candidates`, `route_pos`, `true_arc_id`, `align_dist_m`, `align_margin_m`, `has_positive`, `label_status`, baseline flags |
| `student_test_candidates.parquet` | 209,584 | 13.0 MB | same schema, **separate partition** |
| `student_test_points.parquet` | 19,160 | 0.8 MB | same schema |
| `train/…`, `student_test/…` | per trip | — | per-trip shards (resume and per-trip loading for the Phase 6 HMM) |

Training-ready subset (labelled points only): **1,824,953 rows, 185,161 points, positive rate 10.15 %**.
Rows per trip: min 2,575 / median 18,749 / max 92,740. Candidates per point peak at 150 (trip 58).
The whole `data/processed/phase3/` folder is 267 MB. It sits inside OneDrive, so it will sync.

---

## 6. Runtime & memory

**Full build** (3 workers, one fresh process per trip): train **85.2 s wall**, student-test 10.2 s, **108.5 s total** including consolidation. Per-trip peak memory is **≤ 396 MB** (trip 13, 307 k arcs); the baseline after imports is 108 MB. The consolidation step streams shards and never holds all trips at once.

**Per stage** (benchmark A, median of 3 runs, fresh process each):

| Trip | Points | Arcs | Load | Candidates | Features | Labels | Total | Inference only | Peak MB |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 00000026 | 361 | 5,043 | 0.04 | 0.005 | 0.03 | 0.01 | 0.09 s | 0.08 s | 119 |
| 00000001 | 1,315 | 165,790 | 1.36 | 0.17 | 0.09 | 0.11 | 1.87 s | 1.71 s | 268 |
| 00000013 | 2,580 | 307,392 | 2.29 | 0.29 | 0.13 | 0.28 | 3.19 s | 2.84 s | 392 |
| 00000054 | 19,313 | 116,916 | 1.60 | 0.15 | 0.21 | 1.01 | 3.08 s | 2.05 s | 229 |
| 00000092 | 7,892 | 53,348 | 0.67 | 0.13 | 0.38 | 2.30 | 3.58 s | 1.23 s | 243 |

Candidates plus features take ≤ 0.6 s on every trip. The `load_trip` text parser from Phase 2 dominates on big networks (reused unchanged). Loading the full training table for Phase 4 takes 1.3 s, a 370 MB DataFrame and ~800 MB process peak.

---

## 7. Tests — 29 / 29 passing

Geometry (vectorised vs scalar reference, degenerate segments, angle wrap, bearing convention, projection vs haversine). Candidates (random 600-arc network vs brute force at 30/40/50 m, long-arc regression, sorting and cap, empty inputs, config validation, real trip vs brute force plus trip-local arc IDs). Features (kinematics and gap split, heading carry-forward, directional and connectivity features on a two-way road, label-free feature list, AST check that features never touch the route, **features bit-identical with and without the route**). Labels (many points per arc plus skipped short arcs, U-turn direction, revisited arcs, off-route tail, multi-edge twin). Partitions and built dataset (80/10 disjoint, ≤ 1 positive per point, no duplicate rows, no nulls in features, loader filter).

---

## 8. Data-quality findings & corrections to Phase 2 docs

- **Route connectivity**: Phase 2's doc says 89/90 trips are 100 % connected. Phase 2 only flagged trips below 99 %. In fact **11 trips have at least one break**: train 02, 19, 33, 41, 42, 69, 75, 80, 86, 89 (98.13–99.93 %) and student-test 76 (99.73 %).
- **Phase 2's "< 0.5 % miss rate"** was measured on trip 1 only. Over all 80 train trips at 50 m it is 1.79 % pooled (excluding route-end points) with a 0.00 % per-trip median. The worst trip is 54 (25.3 %).
- Time gaps: up to 153,064 s (trip 90) and 52,810 s (trip 78). Gaps over 30 s split the track into 213 segments (train).
- Sustained 85–140 m/s GPS speeds on smooth paths in trip 13. That looks like a timing artifact, so speeds are not hard-filtered.
- Duplicate multi-edge arcs (same from/to nodes): 1,775 in 53 of the 80 train networks (max 322 in one trip). Across all 90 trips, 11 of 34,813 route entries use one.
- The point count is 208,417 (raw files and Phase 2 JSON agree). `DATASET_EXPLORATION.md` says 208,416, which is a typo.

## 9. Remaining issues / hand-off to Phase 4
1. **Train only on labelled points** (`load_candidates(..., labelled_only=True)`). Consider `drop_boundary=True` as an ablation, since 12 % of labelled points sit on junction boundaries.
2. **Class imbalance**: about 1 positive per 9.9 candidates. Top-1 per point (`groupby point → argmax`) is the primary metric, not row accuracy.
3. **Heavy trips**: trips 54, 84 and 21 hold 24.5 % of all train GPS points (15.8k–19.3k each). Consider per-trip weights so a few trips do not dominate a fold. Only trip 54's extra points are mostly unlabelled.
4. **Ambiguity floor**: 6.6 % of labelled points have a negative that is indistinguishable at that instant (junction nodes). The temporal and connectivity features, and Phase 6's HMM, are what can resolve them.
5. Off-route points are unlabelled (11.16 % of train points; 2.87 % excluding trip 54), so Top-1 on them is undefined. Phase 6 still needs a route through them (gap handling).
6. `load_trip` (Phase 2) is the slowest stage, at ~1.3–2.3 s on large networks. It is fine for the API, but a vectorised parser would cut it about 10×. This is optional and was not changed in Phase 3.
