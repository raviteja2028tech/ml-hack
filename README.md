# PathSnap — GPS Map-Matching & Route Reconstruction

Matches noisy GPS tracks to the trip's road network. Each GPS point gets candidate road arcs, and a classifier scores them. A planned HMM/Viterbi decoder will turn those scores into a connected route.

## Status
| Phase | Component | State |
|---|---|---|
| 2 | Dataset loading, projection, audit, Folium EDA maps | done |
| 3 | Candidate generation, feature engineering, route labelling | done (`docs/PHASE3_REPORT.md`) |
| 4 | 5-model training and evaluation | done. Best is LightGBM: CV Top-1 0.8885, student-test Top-1 0.8952 |
| 5 | Optuna Bayesian tuning | not started |
| 6 | HMM + Viterbi decoding | not started |
| 7–8 | FastAPI backend, evaluator UI | not started |

## Layout
```
src/
  pathsnap_loader.py       read .track/.nodes/.arcs/.route, local metric projection
  candidate_generator.py   exact radius search for candidate arcs (densified cKDTree)
  feature_engineering.py   42 per-candidate features (distance, heading, speed, topology, temporal)
  route_labeler.py         monotone DP alignment of GPS to .route -> candidate labels
  phase3_pipeline.py       per-trip candidates + features + labels
  phase3_dataset.py        load the processed Parquet dataset
  train_models.py          GroupKFold-by-trip training/evaluation of 5 models
  visualize_folium.py      multi-layer trip maps
scripts/
  run_audit.py             dataset audit -> outputs/reports/dataset_exploration_report.json
  generate_eda_maps.py     Folium maps -> outputs/figures/eda_maps/
  generate_review1_map.py  outputs/figures/REVIEW1_MAP.html
  build_phase3_dataset.py  builds data/processed/phase3/ (about 2 min, 3 workers)
  benchmark_phase3.py      runtime/memory benchmark
tests/                     unittest suites for phases 3 and 4
models/phase4/             trained models (joblib bundles: model + feature list)
outputs/reports/           audit, Phase 3 and Phase 4 results (JSON + logs)
docs/                      requirements, plan, progress, reports, official guidelines
```

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Extract the official `Student folder.zip` into `data/raw/`, so that `data/raw/Student folder/train/` and `data/raw/Student folder/student_test/` exist. Raw and processed data are not tracked in Git.

## Run
```powershell
.\.venv\Scripts\python.exe scripts\run_audit.py
.\.venv\Scripts\python.exe scripts\generate_eda_maps.py
.\.venv\Scripts\python.exe scripts\build_phase3_dataset.py          # regenerates data/processed/phase3
.\.venv\Scripts\python.exe -m src.train_models --smoke              # quick check
.\.venv\Scripts\python.exe -m src.train_models                      # all 5 models (about 14 min)
.\.venv\Scripts\python.exe -m unittest tests.test_phase3 tests.test_phase4
```
