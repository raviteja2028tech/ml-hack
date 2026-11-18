# PathSnap: Technical Report (draft)

Every number in this report is read from files in `results/metrics/`, `reports/` and `outputs/reports/` that the code in this repository produced.

## 60-second explanation
> GPS points are noisy, so the nearest road is right only about half the time (53.6% on the student-test trips). For each GPS point we gather every road arc within 50 metres. A LightGBM classifier then scores each candidate using 42 features: distance, heading agreement, speed, road topology, and whether the arc stays consistent with neighbouring points. On its own, that picks the correct road for 89.5% of points.
>
> Roads must connect, though. So we treat the candidates as hidden states of a Hidden Markov Model. The classifier's probabilities are the emissions. The transitions compare the driving distance through the directed road network with the distance the GPS actually moved. Viterbi then finds the single most plausible connected route.
>
> On the 10 held-out trips this raises point accuracy to 94.2% and route continuity to 99.9%, and the HMM wins on every trip. The FastAPI server does this for any trip in 0.2–2.8 s, and the dashboard on a second laptop shows the raw GPS, both predictions and the ground truth.

## 1. Data
- 80 training trips (208,417 GPS points) and 10 student-test trips (19,160 points).
- Each trip has its own road network (trip-local ids), a 1 Hz track, and a reference route given as an arc sequence.
- See `reports/DATASET_EXPLORATION.md` and `reports/PHASE3_REPORT.md`.

## 2. Candidates, features, labels (Phase 3)
- **Candidate search:** an exact radius search (R = 50 m) on densified arcs. It is 89–178× faster than brute force with zero misses. The original midpoint KD-tree found only 47.7–63.4% of the true candidates.
- **Candidate volume:** 1,869,503 candidate rows for the training trips, about 9 per point.
- **Labels:** produced by a monotone DP alignment of the track to `.route`. 88.84% of training points are labelled. Points where the vehicle left the reference route are flagged `off_route` and excluded.
- **Leakage control:** features are computed with `trip.route = None`. A test checks that features are bit-identical with and without the route, and an AST test checks that the feature module never reads the route.

## 3. Classifiers (Phase 4)
- **Protocol:** 5-fold GroupKFold by trip on the 80 training trips (trip-balanced CV subset), then a final fit on all 1,824,953 labelled training rows.
- **Selection:** by CV Top-1. Student-test was not used to choose.

| Model | CV Top-1 | Student-test Top-1 | Test F1 | Test log loss |
|---|---|---|---|---|
| Logistic Regression | 87.49% ± 1.76 | 87.93% | 0.8734 | 0.0615 |
| Random Forest | 87.59% ± 1.50 | 88.22% | 0.8811 | 0.0521 |
| Extra Trees | 87.10% ± 1.55 | 87.71% | 0.8674 | 0.0625 |
| **LightGBM** (selected) | **88.85% ± 1.35** | **89.52%** | **0.8898** | **0.0482** |
| MLP | 88.23% ± 1.39 | 88.67% | 0.8823 | 0.0531 |

## 4. HMM + Viterbi (Phase 5)
- The full definition is in `docs/ARCHITECTURE.md`.
- **Emissions:** log LightGBM probability.
- **Transitions:** directed-network distance vs GPS displacement (β = 10 m), plus speed, backward-move and U-turn penalties.
- **Disconnected candidates:** a finite penalty, so decoding never fails on map gaps.
- **Decoding:** exact Viterbi with back-pointers.
- **Chains:** split at time gaps > 30 s.
- **Tuning:** none. The parameters were fixed before evaluation.

## 5. ML-only vs HMM (student-test, 19,133 labelled points; `results/metrics/hmm_evaluation.json`)
| Metric | ML-only | HMM |
|---|---|---|
| Top-1 point accuracy | 89.52% | **94.15%** |
| Continuity, direct | 95.80% | **98.31%** |
| Continuity, ≤ 3 arcs | 98.53% | **99.92%** |
| Arc switches | 4,191 | 3,229 |
| Mean route arc precision | 91.01% | **97.56%** |
| Mean route arc recall | 89.68% | **90.77%** |
| Mean route length F1 | 94.76% | **98.44%** |

The reference labels themselves score 99.94% on the ≤ 3-arc continuity measure, so the HMM output is nearly as connected as the ground truth.

Per trip (inference time covers loading, candidates, features, LightGBM and Viterbi):

| Trip | Points | ML-only | HMM | Inference |
|---|---|---|---|---|
| 00000000 | 2503 | 95.21% | 96.04% | 965 ms |
| 00000007 | 1354 | 97.64% | 99.63% | 396 ms |
| 00000009 | 1049 | 87.70% | 91.04% | 1783 ms |
| 00000040 | 2995 | 94.42% | 97.20% | 1167 ms |
| 00000052 | 262 | 96.18% | 98.85% | 170 ms |
| 00000056 | 3958 | 91.17% | 96.62% | 2784 ms |
| 00000065 | 768 | 90.76% | 93.36% | 868 ms |
| 00000076 | 1486 | 92.46% | 96.64% | 2801 ms |
| 00000077 | 3911 | 76.20% | 86.17% | 2426 ms |
| 00000091 | 874 | 90.16% | 93.14% | 1100 ms |

**Demonstration trip 00000052** (262 points):
- The HMM changes 8 points at a junction where the ML-only argmax briefly jumps to a parallel road.
- Accuracy rises from 96.18% to 98.85%.
- Route arc precision rises from 91.76% to 100%.

## 6. Bayesian optimisation (Optuna)
Source: `results/metrics/optuna_lightgbm.json` and `.trials.csv`.
- **Setup:** Optuna TPE, 12 trials, 254 s, scored with GroupKFold(3) by trip over all 80 training trips (cap of 300 labelled points per trip, 295,527 candidate rows). Student-test was never loaded.
- **Baseline:** trial 0 is the deployed Phase 4 configuration, scored with the same protocol: **88.76%** CV Top-1.
- **Best:** trial 2 reached **89.00%** (+0.24 pts) with `n_estimators=750, learning_rate=0.080, num_leaves=111, min_child_samples=21, subsample=0.98, colsample_bytree=0.90, reg_lambda=0.007`.
- **Decision:** the gain is small and comes from a reduced 3-fold subset, so the deployed model was **not** replaced. Adopting it would need a full retrain and confirmation under the 5-fold protocol.

## 7. System (Phases 6–9)
- **Pipeline:** `src/phase6_pipeline/end_to_end_matcher.py` (CLI included).
- **API:** `src/phase7_api/app.py`, with endpoints `/health`, `/trips`, `/predict` and `/metrics`. The model loads once at startup, `/predict` results are cached per trip, trip ids are validated, and errors never expose local paths.
- **Dashboard:** `src/phase8_ui/`.
- **Two-laptop setup:** `docs/DEMO_GUIDE.md`.

## 8. Limitations
- Student-test is a development set; the evaluator's hidden test set has not been seen.
- The HMM parameters are hand-set (β = 10 m and so on). Tuning them on the training trips with out-of-fold emissions is future work.
- The hardest trip, 00000077, still reaches only 86.17%. Its error pattern has not been analysed yet.
- The map tiles and Leaflet need internet access in the browser.

## 9. Requirement checklist
| Requirement | Status |
|---|---|
| Data loading, EDA, Folium maps | done |
| Candidate generation + feature engineering + leakage-safe labels | done |
| 5 models, grouped CV, comparison | done |
| Bayesian optimisation (Optuna) | done (bounded budget; see §6) |
| HMM + Viterbi decoder | done, tested |
| End-to-end pipeline + CLI | done, tested on all 10 student-test trips |
| FastAPI `/health /trips /predict /metrics` | done, tested (TestClient + live HTTP) |
| Dashboard with map, layers, metrics, latency | done, verified in a headless Chrome |
| Two-laptop LAN demo | server binds 0.0.0.0 and answers on the LAN IP; **second-laptop test not performed** |
| PDF technical report | **not generated**; this Markdown is the draft |
