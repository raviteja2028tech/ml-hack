# PathSnap: Judge Presentation & Viva Defense Script

---

## A. 30-Second Introduction (Elevator Pitch)
> "Good morning, Judges. We present **PathSnap**, a distributed, real-time map-matching and route reconstruction system. In urban environments, raw GPS signals suffer from severe multipath noise, signal jumps, and drift. PathSnap takes noisy, raw GPS trajectories and maps them back to the underlying road network graph. We combine a high-speed **LightGBM classifier** with a **Hidden Markov Model and Viterbi decoder** to achieve **94.15% pointwise accuracy** and **99.92% route continuity** across real-world driving trips in under 200 milliseconds per trip."

---

## B. 60-Second Technical Summary
> "Our architecture operates in four distinct stages:
> 1. **Geospatial Projection & Candidate Generation**: We project angular coordinates to local metric space and use a densified KD-Tree within a 50-meter radius to find candidate road segments in sub-millisecond time.
> 2. **Feature Engineering**: We extract 42 kinematic and topological features per candidate—including perpendicular distance, bearing difference, speed consistency, and network degree—without any data leakage.
> 3. **Machine Learning Emission Scoring**: We trained and compared five distinct algorithm families. LightGBM proved superior, achieving 89.52% Top-1 accuracy on unseen test data.
> 4. **Sequential HMM / Viterbi Decoding**: Pointwise predictions can still produce disconnected jumps on parallel roads. Our HMM models road-network graph transitions, penalizing impossible speeds and U-turns. Viterbi decoding solves for the global maximum-likelihood route, lifting accuracy to 94.15% and route continuity to 99.92%.
> The system is deployed via FastAPI on Laptop 1 and served to an evaluator dashboard on Laptop 2 over LAN."

---

## C. 2-Minute Live Demonstration Script

| Time | Action | What to Say |
| :--- | :--- | :--- |
| **0:00 - 0:25** | Open `http://localhost:8000/ui/` in browser. Show top bar status: `API: OK, Model: LightGBM`. | *"Here is our live evaluator dashboard. Notice the connection status shows our FastAPI inference server is online with the pre-loaded LightGBM model. The UI displays the 10 official student-test trips."* |
| **0:25 - 0:55** | Select Trip `00000052` (Short) and click **Run Map Matching**. | *"Let's run Trip 00000052. Within 160 milliseconds, the server generates candidates, computes 42 features, runs LightGBM, and decodes the HMM. On the map: blue dots are raw GPS points; orange is ML-only; green is our reconstructed HMM route; purple is reference ground-truth."* |
| **0:55 - 1:25** | Zoom into an intersection where HMM fixed an ML error. Point out the pink/red dot (`Points changed by HMM`). | *"Notice here: at this complex intersection, raw GPS drifted toward a parallel street. The ML-only model (orange) was tempted by proximity, creating a disconnected jump. But the HMM transition model recognized that road connectivity made that jump impossible, choosing the true connected road (green), perfectly matching ground truth."* |
| **1:25 - 2:00** | Select a longer trip (`00000040` - 2,995 points) and click Run. Show the metrics table. | *"Now let's test Trip 00000040—nearly 3,000 GPS points. In just 720 milliseconds, it processes the entire 45-minute trip. The metrics table shows ML Top-1 accuracy at 94.42%, which HMM boosts to 97.20%, and reaches 100% route continuity."* |

---

## D. The Five Machine Learning Models Evaluated

In Phase 4, we trained and cross-validated five distinct algorithm families on the 80 training trips (208,000+ points) using 5-fold `GroupKFold` grouped by `trip_id`:

| Model | Algorithm Family | CV Top-1 Accuracy | Student-Test Top-1 | Test Macro F1 | Test Continuity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **LightGBM** | Histogram Gradient Boosting | **88.85%** | **89.52%** | **0.9013** | **98.53%** |
| **Random Forest** | Ensemble Bagging Trees | 88.01% | 88.64% | 0.8924 | 98.12% |
| **Extra Trees** | Extremely Randomized Trees | 87.89% | 88.42% | 0.8901 | 98.05% |
| **MLP** | Multi-Layer Perceptron (Neural) | 85.12% | 85.91% | 0.8643 | 96.88% |
| **Logistic Regression** | Linear / Probabilistic | 82.40% | 83.15% | 0.8350 | 95.20% |

---

## E. Why LightGBM Was Selected
1. **Highest Cross-Validation & Test Top-1 Accuracy**: Outperformed all other families on both 5-fold GroupKFold CV (88.85%) and unseen test trips (89.52%).
2. **Computational Efficiency**: Trained in only 1.8 seconds on CPU and performs inference at **< 0.15 ms per GPS point**, essential for real-time mobile deployment.
3. **Robust Handling of Non-Linear Geospatial Interactions**: Effectively models interactions between distance, heading angles, and velocity without requiring complex manual feature scaling.

---

## F. Why HMM Was Needed After ML
Machine learning classifiers treat each GPS observation independently. However:
- Two consecutive GPS points might be closest to two road segments that have **no physical road connection** (e.g. overpasses, parallel one-way streets, or opposing highway lanes).
- An independent classifier has no concept of vehicle momentum or road network graph topology.
- **The HMM bridges this gap**: It enforces that transitions between road segments must be physically drivable in the underlying road network graph.

---

## G. How Viterbi Decoding Works
1. **States**: At each time step $t$, the hidden states are the spatial candidate road segments $\{s_1, s_2, \dots, s_K\}$.
2. **Emissions**: $V_t(j) = \log P(z_t \mid s_j) + \max_i [V_{t-1}(i) + \log P(s_j \mid s_i)]$.
   - Emission probability combines the LightGBM candidate confidence with Gaussian GPS measurement distance decay: $P(z_t \mid s_j) \propto P_{\text{ML}}(s_j) \cdot \exp(-d^2 / 2\sigma_z^2)$.
3. **Transitions**: $\log P(s_j \mid s_i)$ evaluates the road-network shortest path distance between the end of arc $s_i$ and the start of arc $s_j$ compared to the Euclidean distance traveled between GPS points. Disconnected jumps and speed violations receive large negative log-penalties.
4. **Backtracking**: At the final GPS point $T$, Viterbi backtracks through the stored backpointer trellis to reconstruct the globally optimal, continuous sequence of road arcs.

---

## H. Top-1 Point Accuracy vs Route Continuity
- **Top-1 Observation Accuracy**: The percentage of individual GPS points where the assigned road segment matches the ground-truth traversed segment.
- **Route Continuity (Reach-3)**: The percentage of consecutive road segments in the reconstructed route that are topologically connected within 3 graph hops in the directed road network. A human driver cannot teleport between disconnected roads; continuity measures whether the output is a physically realistic path.

---

## I. Why Accuracy Improved from 89.52% to 94.15%
- LightGBM alone gets ~89.5% accuracy because ~10.5% of GPS points suffer from multipath reflections that place the raw point closer to a wrong road (e.g., a parallel alleyway or cross street).
- The HMM corrects these points because the wrong road segment has zero or high-penalty transition connectivity to the points before and after it.
- By enforcing graph connectivity, the Viterbi algorithm overrides local noisy GPS deviations, correcting **~3% to 4% of all trajectory points** and eliminating 962 spurious road switches.

---

## J. Top 10 Judge Questions & Model Answers

### Q1: "Did you use ground-truth labels during candidate generation or feature extraction?"
> **Answer**: "No. All candidate generation, feature engineering, and HMM decoding are 100% label-free. The reference `.route` file is loaded strictly server-side inside `evaluate=True` to compute accuracy and display the comparison on the dashboard. In a blind production deployment, `evaluate=False` runs identically without touching reference data."

### Q2: "Why use local equirectangular metric projection instead of Euclidean degrees?"
> **Answer**: "Latitude and longitude are angular coordinates on an oblate spheroid. At 50° North latitude, one degree of longitude is only ~71.7 km, whereas one degree of latitude is ~111.2 km. Calculating Euclidean distances in degrees creates an elliptical distortion of over 35%. Our metric projection computes accurate distances in physical meters."

### Q3: "What happens if a vehicle stops at a red light?"
> **Answer**: "When speed drops below 0.5 m/s, vehicle bearing becomes unstable due to GPS jitter. Our kinematics module explicitly detects stationary points and carries forward the last valid moving heading, preventing random directional errors from corrupting candidate bearing features."

### Q4: "What is Candidate Miss Rate and what was yours?"
> **Answer**: "Candidate Miss Rate is the fraction of GPS points where the true road segment is missing from the candidate set. With our densified 50-meter KD-Tree radius, our miss rate across all 19,133 student-test points is **under 0.3%**, ensuring the downstream ML model almost always has the correct segment in its candidate pool."

### Q5: "How does your HMM handle large GPS dropouts (e.g. tunnels)?"
> **Answer**: "If the time gap $\Delta t$ between consecutive observations exceeds 30 seconds, our HMM recognizes that sequential correlation is broken. It restarts the Viterbi chain with uniform prior probabilities rather than penalizing the long gap, preventing numerical underflow."

### Q6: "Why LightGBM instead of Deep Learning (e.g., LSTM or Transformer)?"
> **Answer**: "Our tabular candidate features are heterogeneous (distances, angles, graph degree, speed ratios). Gradient-boosted decision trees naturally handle unnormalized tabular features, train in seconds on CPU, and run inference in microseconds. Sequence modeling is handled by the road-network HMM, which strictly respects physical graph topology that a neural net might violate."

### Q7: "How did you prevent data leakage during model training?"
> **Answer**: "We utilized `GroupKFold` grouped strictly by `trip_id` across the 80 training trips. All GPS points from a single vehicle trip belong entirely to either the training fold or the validation fold. Scalers and feature transformers are fitted solely on training folds."

### Q8: "Can your system run completely offline?"
> **Answer**: "Yes. The backend runs 100% offline with zero cloud APIs. The model weights are bundled locally as `.joblib` files, the Leaflet mapping libraries are bundled locally in `src/phase8_ui/vendor/`, and the server runs locally on `0.0.0.0:8000`."

### Q9: "How does the two-laptop deployment work?"
> **Answer**: "Laptop 1 runs Uvicorn on `0.0.0.0:8000`. Laptop 2 connects to `http://<Laptop1_IP>:8000/ui/` across the local Wi-Fi router or hotspot. The web frontend sends REST requests across LAN, and the backend processes map matching and returns JSON."

### Q10: "What is the biggest limitation of your current approach?"
> **Answer**: "Our candidate generator uses a fixed 50m radius. In sparse rural highways with very long arcs or wide GPS blunders (>50m), candidates can be missed. An adaptive radius based on vehicle speed and GPS Dilution of Precision (HDOP) would further improve robustness."
