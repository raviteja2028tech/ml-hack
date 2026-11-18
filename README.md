# PathSnap: GPS Map Matching & Route Reconstruction

PathSnap snaps noisy GPS tracks onto the trip's road network. For every GPS point it:
1. generates candidate road arcs within 50 m;
2. scores each candidate with a LightGBM classifier (42 features);
3. decodes the most plausible connected road sequence with an HMM + Viterbi.

A FastAPI backend serves predictions to a Leaflet dashboard, which can run on a second laptop.

## Results (10 student-test trips, 19,133 labelled GPS points)
| Method | Top-1 point accuracy | Continuity (≤3 arcs) | Route arc precision | Route length F1 |
|---|---|---|---|---|
| Nearest road (heuristic) | 53.64% | – | – | – |
| Distance + heading (heuristic) | 85.75% | – | – | – |
| LightGBM only (per-point argmax) | 89.52% | 98.53% | 91.01% | 94.76% |
| **LightGBM + HMM/Viterbi** | **94.15%** | **99.92%** | **97.56%** | **98.44%** |

- The HMM improves Top-1 on all 10 trips; full numbers are in `results/metrics/hmm_evaluation.json`.
- The HMM settings were fixed before evaluation, not tuned on these trips.
- The model was chosen among 5 classifiers by grouped 5-fold CV on the 80 training trips (`results/metrics/model_comparison.json`).

## Project layout
```
src/
  phase1_eda/            data loading, metric projection, Folium maps, dataset audit
  phase2_preprocessing/  geodesic helpers
  phase3_features/       candidate generation, 42 features, route labelling, dataset builder
  phase4_models/         5-model training/evaluation, Optuna Bayesian optimisation
  phase5_hmm/            transition model, Viterbi decoder, ML-only vs HMM evaluation
  phase6_pipeline/       end-to-end inference for one trip (CLI)
  phase7_api/            FastAPI backend (/health /trips /predict /metrics)
  phase8_ui/             dashboard (HTML + Leaflet, no build step)
api/main.py              backward-compatible ASGI entry point (imports src.phase7_api.app)
artifacts/models/        trained models (LightGBM is deployed)
results/metrics/         model comparison, HMM evaluation, Optuna results
results/maps/            Folium maps
tests/                   68 unittest tests
docs/                    architecture, demo guide, requirements, guidelines
reports/                 technical report draft, Phase 3 report, logs
```

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
Extract the official `Student folder.zip` into `data/raw/`, so that `data/raw/Student folder/train/` and `data/raw/Student folder/student_test/` exist. Raw and processed data are not tracked in Git.

## Run the demo
**Laptop 1, backend:**
```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```
To use it on one laptop, open **http://localhost:8000/ui/**.

**Laptop 2, frontend:** see `docs/DEMO_GUIDE.md`. Either open `http://<LAPTOP1_IP>:8000/ui/` directly, or serve the UI locally:
```powershell
python -m http.server 5500 --directory src\phase8_ui
# then open http://localhost:5500/?api=http://<LAPTOP1_IP>:8000
```

## Command line
```powershell
.\.venv\Scripts\python.exe -m src.phase6_pipeline.end_to_end_matcher 00000052        # one trip, prints metrics
.\.venv\Scripts\python.exe -m src.phase5_hmm.evaluate_hmm                           # all 10 student-test trips
.\.venv\Scripts\python.exe -m src.phase4_models.bayesian_optimization --trials 12   # Optuna (train trips only)
.\.venv\Scripts\python.exe scripts\build_phase3_dataset.py                          # regenerate data/processed/phase3
.\.venv\Scripts\python.exe -m src.phase4_models.train_models                        # retrain the 5 models (~14 min)
```

## Tests
```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_structure tests.test_phase3 tests.test_phase4 tests.test_phase5_hmm tests.test_phase6_7_pipeline_api
```
