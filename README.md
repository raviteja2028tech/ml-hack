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
PathSnap/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── Student folder/          # 80 train + 10 student_test trips
│   └── processed/
│       ├── train_candidates.parquet
│       └── student_test_candidates.parquet
├── src/
│   ├── phase1_eda/                  # Data loading, projections, Folium visualizations
│   ├── phase2_preprocessing/        # Spatial indexing & coordinate conversions
│   ├── phase3_features/             # Candidate generation, 42 features, route labeling
│   ├── phase4_models/               # 5-model training, evaluation & metrics
│   ├── phase5_hmm/                  # HMM & Viterbi sequential decoder
│   └── phase6_pipeline/             # Unified end-to-end inference pipeline
├── api/                             # FastAPI service (Laptop 1 server)
├── ui/                              # Evaluator interface (Laptop 2 client)
├── artifacts/
│   └── models/                      # Trained .joblib models (LR, RF, ET, LGBM, MLP)
├── results/
│   ├── figures/                     # Confusion matrices & evaluation plots
│   ├── metrics/                     # Model comparison JSONs & benchmarks
│   └── maps/                        # Interactive Folium maps (REVIEW1_MAP.html)
├── tests/                           # Unit tests (45 tests covering all phases)
├── docs/                            # Official guidelines & hackathon requirements
└── reports/                         # Technical reports, EDA summaries & build logs
```

## Setup
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Extract the official `Student folder.zip` into `data/raw/`, so that `data/raw/Student folder/train/` and `data/raw/Student folder/student_test/` exist. Raw and processed data are not tracked in Git.

## Testing
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_structure tests.test_phase3 tests.test_phase4
```

## Run
```powershell
.\.venv\Scripts\python.exe scripts\run_audit.py
.\.venv\Scripts\python.exe scripts\generate_eda_maps.py
.\.venv\Scripts\python.exe scripts\build_phase3_dataset.py       # regenerates data/processed/phase3
.\.venv\Scripts\python.exe -m src.phase4_models.train_models     # all 5 models (about 14 min)
```
