# PATHSNAP: DISTRIBUTED MAP-MATCHING & ROUTE RECONSTRUCTION
## Official Hackathon Requirements & Specification Matrix

**Course**: UE24CS352A — Machine Learning  
**Department**: Computer Science and Engineering, PES University RR Campus  
**Challenge Title**: PathSnap: Distributed Map-Matching & Route Reconstruction  

---

## 1. Problem Statement & Challenge Overview
GPS tracking devices produce noisy, inaccurate location measurements due to urban canyons, signal interference, multipath reflections, satellite dropouts, and low sampling rates. A vehicle traveling on a road network frequently appears to jump onto parallel streets, cut through buildings, or deviate from reality.

**Objective**: Build **PathSnap**, an end-to-end, distributed machine learning system that takes noisy raw GPS trajectory observations, matches them against a discrete road network, reconstructs the true sequential vehicle trajectory using ML classification and Hidden Markov Model (HMM) Viterbi decoding, and serves the results across a two-laptop distributed network with interactive map visualizations.

---

## 2. Distributed Two-Laptop Deployment Architecture

| Role | Machine | Assigned Components & Responsibilities |
| :--- | :--- | :--- |
| **Server & Inference Node** | **Laptop 1** | Hosts FastAPI backend service, trained ML classifiers, road-network graph topologies, HMM/Viterbi decoder engine, and internal server-side ground-truth evaluation logic. Terminal logs must display incoming inference requests. |
| **Client & Evaluator Interface** | **Laptop 2** | Evaluator-facing web interface (React + Vite). Lists the 10 evaluation trips, triggers inference via REST API across LAN, displays predicted routes, ground truth comparison, and interactive Folium/Leaflet map visualizations. |

> **Evaluation Protocol**:
> 1. Evaluator opens UI on Laptop 2.
> 2. Evaluator selects any of the 10 reserved evaluation trips.
> 3. Laptop 2 transmits the selected `trip_id` to Laptop 1 (`POST /predict` or `POST /evaluate`).
> 4. Laptop 1 executes candidate generation -> ML inference -> HMM/Viterbi decoding -> server-side metric computation.
> 5. Evaluator inspects live request logs on Laptop 1's console.
> 6. Laptop 2 renders the predicted route, metrics, and multi-layer map.

---

## 3. Dataset Specifications & Partitions

### 3.1 Partitions
- **Training Set**: 80 trips (`data/raw/Student folder/train/00000001` to `...00000080`). Used for feature engineering, model training, cross-validation, and Optuna Bayesian tuning.
- **Student-Testing Set**: 10 trips (`data/raw/Student folder/student_test/`). Used for development-stage model comparisons, HMM tuning, and validation.
- **Evaluator-Testing Set**: 10 trips (Provided during live final evaluation). Kept completely isolated on Laptop 1; never exposed to student-side training or client UI.

### 3.2 File Formats Per Trip
Every trip folder contains four headerless whitespace-separated files:
1. `.track` — Raw GPS observations:
   - Column 0: Longitude (float, angular degrees)
   - Column 1: Latitude (float, angular degrees)
   - Column 2: Timestamp (integer seconds). Nominally 1 Hz (~97% of steps are exactly 1s apart).
2. `.nodes` — Road network intersection vertices:
   - Line index (0-based) represents `node_id`.
   - Column 0: Longitude (float)
   - Column 1: Latitude (float)
3. `.arcs` — Road network directed segments:
   - Line index (0-based) represents `arc_id`.
   - Column 0: `from_node_id` (integer)
   - Column 1: `to_node_id` (integer)
4. `.route` — Ground-truth vehicle trajectory:
   - 1 integer column representing the chronological sequence of traversed `arc_id` values.
   - *Note*: One GPS point does not map 1:1 to one route arc. A vehicle may emit multiple GPS points while traversing one arc, or skip GPS points across short arcs.

---

## 4. Modeling & Machine Learning Requirements

### 4.1 Candidate Generation & Geospatial Feature Engineering
For each GPS observation:
- Spatial search radius to identify candidate road segments (`arc_id`s within $R$ meters).
- Calculate geodesic / projected distances (Haversine or UTM meters, never naive Euclidean degrees).
- Compute geospatial features:
  - Perpendicular distance from GPS point to candidate arc segment.
  - Angular difference between GPS bearing (heading derived from $\Delta t$) and arc orientation.
  - Velocity consistency ($d/\Delta t$ vs typical road speed limit).
  - Temporal and spatial displacement between consecutive GPS points.
  - Road network topological connectivity from previous candidates.

### 4.2 Minimum 5 Model Requirement
Must train and evaluate at least 5 distinct algorithms across multiple families:
1. **Linear / Probabilistic**: Logistic Regression (L2 regularized)
2. **Ensemble Trees**: Random Forest Classifier
3. **Randomized Trees**: Extra Trees Classifier
4. **Gradient Boosting**: HistGradientBoosting / LightGBM (CPU histogram boosting)
5. **Neural Network**: Multi-Layer Perceptron (MLPClassifier)

### 4.3 Mandatory Evaluation Metrics
- **Observation-level Top-1 Accuracy**: Fraction of GPS points where the top-scoring candidate matches the reference road segment.
- **Candidate Miss Rate**: Fraction of GPS points where the true reference arc was omitted by spatial candidate generation.
- **Macro F1-Score, Precision, Recall**: Multi-class or binary candidate classification quality.
- **Log Loss**: Probabilistic calibration quality for emission estimation.
- **Route Continuity**: Proportion of consecutive predicted segments that share a valid topological road network connection (`to_node_id == next_from_node_id` or reachable).
- **Inference Time & Training Time**: Computational efficiency per trip and per batch.

### 4.4 Bayesian Hyperparameter Optimization
- Tool: `Optuna`
- Strategy: Tree-structured Parzen Estimator (TPE)
- Validation: Group-aware Cross-Validation (`GroupKFold` grouped by `trip_id` across 80 training trips to avoid temporal/spatial data leakage).
- Record: Search space, trial history, objective value, best parameters, and before-vs-after comparison table.

---

## 5. Sequential Route Reconstruction (HMM + Viterbi)

1. **Hidden States**: Candidate road arcs ($s_i \in \text{Candidates}(t)$).
2. **Emission Probabilities**: $P(z_t \mid s_i) \propto \text{ML\_Probability}(s_i \mid z_t) \times \exp\left(-\frac{\text{dist}(z_t, s_i)^2}{2\sigma_z^2}\right)$.
3. **Transition Probabilities**: $P(s_j \mid s_i) \propto \exp\left(-\frac{|\text{network\_dist}(s_i, s_j) - \text{euclidean\_dist}(z_t, z_{t+1})|}{\beta}\right)$.
   - Valid topological transitions receive positive probability; disconnected jumps receive heavy penalties or bridge routing.
4. **Viterbi Decoding**: Dynamic programming on log-probabilities to compute the global maximum-likelihood path across the entire trip.
5. **Comparison**: Report pointwise accuracy, sequence accuracy, and route continuity for Classifier-Only vs Classifier + HMM/Viterbi.

---

## 6. API & Frontend Visualization Requirements

### 6.1 FastAPI Backend (Laptop 1)
- `GET /health` — Service health and loaded model status.
- `GET /trips` — List available evaluation trips.
- `POST /predict` — Run end-to-end inference for a trip ID or GPS stream, returning GeoJSON/JSON of raw points and predicted arcs.
- `POST /evaluate` — Compute Top-1 accuracy, Route Continuity, and F1 using server-side ground truth.

### 6.2 Frontend Interface (Laptop 2)
- Trip selector for the 10 evaluation trips.
- Interactive multi-layer map (Folium / Leaflet):
  - Layer 1: Raw noisy GPS points (blue markers/dots).
  - Layer 2: Predicted route segments (red line).
  - Layer 3: Ground truth reference route (green line, when evaluation mode enabled).
  - Layer 4: Road network background geometry (gray lines).
- Metric dashboard displaying Top-1 Accuracy, Route Continuity, Processing Time, and Model Info.

---

## 7. Mandatory Deliverables & Integrity Policy

1. **GitHub Repository**:
   - Complete data processing, feature engineering, training scripts, Optuna tuning, HMM decoder, FastAPI app, UI code, Folium map generators.
   - Saved model weights and scaler/preprocessor artifacts.
   - Comprehensive README with single-machine and two-laptop LAN run instructions.
2. **Consolidated Technical Report (PDF)**:
   - Section 2.2: Dataset Understanding & EDA (missing values, GPS plots, road network graph stats, Folium maps).
   - Section 3: Model Experimentation (overall 5-model comparison table, individual model subsections with compulsory questions, CV/test comparison charts).
   - Section 4: Bayesian Optimization (search space, objective, trial curves, untuned vs tuned comparison).
   - Section 5: HMM & Viterbi (states, emission/transition math, gap handling, ablation table).
   - Section 6: Folium Geographical Visualizations.
   - Section 7: API, UI, and Deployment screenshots (including Laptop 1 terminal logs).
3. **Academic Integrity & Plagiarism Rules**:
   - No code or weight sharing across teams.
   - Ground-truth evaluation files must stay server-side.
   - Zero fabricated metrics or synthetic training histories.
