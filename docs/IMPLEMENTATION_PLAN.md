# PATHSNAP: MASTER IMPLEMENTATION PLAN

## Overview & Execution Strategy
This document outlines the phased roadmap for building the complete PathSnap Distributed Map-Matching and Route Reconstruction system in the existing `ml-hackthon` workspace on Windows 11.

---

## Phase Breakdown & Work Packages

### Phase 0: Workspace Repair & Baseline Verification (Completed)
- [x] Fix `ModuleNotFoundError: No module named 'agent'` in `agent/local_agent.py` and `scripts/START.ps1`.
- [x] Add workspace path resolution and configure `127.0.0.1` binding.
- [x] Run and verify existing 25-point preflight audit (`scripts/PREFLIGHT.ps1`).
- [x] Run and verify baseline ML pipeline (`scripts/test_pipeline.py`).

### Phase 1: Hackathon Requirements Extraction (Completed)
- [x] Extract `PathSnap_student_materials.zip` to `docs/guidelines/`.
- [x] Extract `Student folder.zip` to `data/raw/` (80 train trips + 10 student-test trips).
- [x] Read and parse all 5 official guidelines: Main Guidelines, Dataset Exploration, Models & Training, Deliverables, Assessment & Plagiarism.
- [x] Compile comprehensive requirement matrix in `docs/HACKATHON_REQUIREMENTS.md`.

### Phase 2: Dataset Exploration & Validation (Immediate Priority)
- [ ] Implement reusable geospatial parser in `src/pathsnap_loader.py`:
  - Read `.track`, `.nodes`, `.arcs`, `.route` for any given trip ID.
  - Project angular coordinates (lon, lat) to local Euclidean metric coordinates (UTM or local Mercator projection) using standard geodesic formulas.
- [ ] Run automated validation across all 80 train trips + 10 test trips:
  - Check missing rows, coordinate bounds, NaN values, sampling interval distribution.
  - Analyze road network graph properties (nodes, directed arcs, degree, disconnected components).
  - Analyze reference route structure vs GPS track timestamps.
- [ ] Generate comprehensive EDA report in `outputs/reports/dataset_exploration_report.json` and `docs/DATASET_EXPLORATION.md`.
- [ ] Generate interactive Folium maps in `outputs/figures/eda_maps/` visualizing sample trips (raw GPS, road network, ground-truth route).

### Phase 3: Geospatial Candidate Generation & Feature Engineering (Completed — see `docs/PHASE3_REPORT.md`)
- [x] Build spatial candidate generator in `src/candidate_generator.py`:
  - For each GPS observation, search within radius $R$ (e.g. 50m) for candidate arcs.
  - Derive ground-truth binary label for each candidate: $y = 1$ if candidate arc equals the ground-truth arc traversed at time $t$, else $y = 0$. (Implemented in `src/route_labeler.py` via monotone DP alignment.)
  - Calculate Candidate Miss Rate.
- [x] Feature extraction module in `src/feature_engineering.py`:
  - Perpendicular distance from GPS point to candidate road segment line.
  - Projection parameter along the segment ($t \in [0, 1]$).
  - Heading / bearing difference between GPS heading vector and road segment direction vector.
  - Speed consistency: segment length vs observed GPS velocity.
  - Network proximity: topological distance from previous candidates.
- [x] Compile training dataset across the 80 training trips with GroupKFold trip IDs (`data/processed/phase3/train_candidates.parquet`).

### Phase 4: Model Training, Evaluation & Benchmarking (5 Models)
- [ ] Implement model training and evaluation harness in `src/train_models.py`:
  1. **Logistic Regression** (Linear baseline)
  2. **Random Forest Classifier** (Ensemble bagging)
  3. **Extra Trees Classifier** (Extremely randomized trees)
  4. **HistGradientBoostingClassifier / LightGBM** (Histogram gradient boosting)
  5. **MLPClassifier** (Neural representation)
- [ ] Metrics computed on 5-fold GroupKFold CV and evaluated on 10 student-test trips:
  - Top-1 Observation Accuracy
  - Macro F1-score, Precision, Recall
  - Log Loss (probability calibration)
  - Route Continuity
  - Training time & Inference latency
- [ ] Export trained models to `models/` and performance logs to `outputs/reports/model_comparison.json`.

### Phase 5: Bayesian Hyperparameter Optimization
- [ ] Build Optuna hyperparameter optimization harness in `src/tune_optuna.py`:
  - Optimize the top-performing model (e.g. HistGradientBoosting / LightGBM or Random Forest).
  - Search space: learning rate, max depth, num leaves, min child samples, regularization.
  - GroupKFold on 80 training trips.
  - 30-50 optimization trials recording trial curves.
- [ ] Save best tuned model artifact to `models/best_tuned_model.joblib`.
- [ ] Compile before-vs-after comparison table.

### Phase 6: HMM + Viterbi Sequential Route Reconstruction
- [ ] Build HMM decoder in `src/hmm_decoder.py`:
  - Hidden states: Candidate road segments for each GPS point.
  - Emission probabilities: Combined ML model output + Gaussian distance penalty.
  - Transition probabilities: Network shortest-path distance vs great-circle GPS displacement.
  - Viterbi algorithm operating in log-space.
  - Handle disconnected segments and GPS gaps.
- [ ] Compare Classifier-Only vs Classifier + HMM across student test trips.
- [ ] Compute route continuity improvements and sequence accuracy.

### Phase 7: FastAPI Backend Service (Laptop 1)
- [ ] Develop `src/api.py`:
  - Load best model, preprocessor, and road network cache.
  - Endpoints: `GET /health`, `GET /trips`, `POST /predict`, `POST /evaluate`.
  - Structured JSON response with raw GPS, predicted route, ground-truth route, and metrics.
  - Console logging of incoming requests for evaluator inspection.

### Phase 8: Web Interface & Map Visualization (Laptop 2)
- [ ] Build React + Vite / Leaflet / Folium web dashboard in `ui/`:
  - Trip selector (10 evaluation trips).
  - Controls to trigger LAN prediction on Laptop 1.
  - Multi-layer map displaying raw GPS, predicted route, reference route.
  - Real-time evaluation metric cards.

### Phase 9: Two-Laptop Network Configuration & Deployment Guide
- [ ] Script and document LAN setup (`scripts/RUN_SERVER.ps1`, `scripts/RUN_CLIENT.ps1`).
- [ ] Document firewall exceptions, host IP binding, and CORS configuration.

### Phase 10: Final Technical Report & Repository Deliverables
- [ ] Generate consolidated PDF technical report adhering to the exact required sections (2 to 7).
- [ ] Include actual Folium map screenshots, model comparison charts, Optuna curves, confusion matrices.
- [ ] Create clean GitHub-ready repository with reproducible README.
