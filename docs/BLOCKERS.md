# PATHSNAP: RISKS, ISSUES & BLOCKERS LOG

**Status**: NO ACTIVE BLOCKERS. Phase 3 data-quality risks logged below (ISS-04 … ISS-07).

---

## Resolved Issues

| ID | Issue Description | Root Cause | Resolution | Status |
| :---: | :--- | :--- | :--- | :---: |
| **ISS-01** | `ModuleNotFoundError: No module named 'agent'` when launching local agent | Python script execution directory (`sys.path[0]`) was set to `agent/`, causing nested module import to fail. | Injected `WORKSPACE_ROOT` into `sys.path` in `agent/local_agent.py` and wrapped fallback imports in `try/except`. | **RESOLVED** |
| **ISS-02** | `AttributeError: 'Agent' object has no attribute 'hooks'` | `run_interactive_loop(agent)` was passed an `Agent` instance rather than `config: AgentConfig`. | Corrected call to `await run_interactive_loop(config)`. | **RESOLVED** |
| **ISS-03** | False `[WARN] Ollama is currently OFFLINE` | Ollama daemon initialization takes ~4s on Windows; script poll timeout was 3s and localhost resolved to IPv6 `::1`. | Updated `START.ps1` to loop-poll for up to 8s using direct IPv4 `http://127.0.0.1:11434`. | **RESOLVED** |

---

## Phase 3 Findings (handled, not blocking)

| ID | Issue | Evidence | Handling | Status |
| :---: | :--- | :--- | :--- | :---: |
| **ISS-04** | Midpoint KDTree misses candidate arcs | Phase 2 `arc_kdtree` recall 47.7–63.4 % vs brute force (benchmark B) | Exact densified-arc index in `src/candidate_generator.py` (0 missing) | **RESOLVED** |
| **ISS-05** | Reference route omits driven sections | Trip 33: 3,042 points (55 min) on mapped roads at a route break; trip 02: 311 points | Labelled `off_route` (no positive); `load_candidates()` drops them by default | **MITIGATED** |
| **ISS-06** | Track continues far beyond the route end | Trip 54: ~16.6k points parked 70–200 m off-network for ~4.6 h | No candidates / `off_route`; miss rates reported with and without route-end points | **MITIGATED** |
| **ISS-07** | Junction ambiguity in labels | 12.0 % of labelled train points `boundary`; 6.6 % have an indistinguishable negative at that instant | Kept and flagged; temporal features + Phase 6 HMM to resolve | **MONITOR** |

## Active Risk Monitoring
1. **Coordinate Projection**: GPS coordinates are spherical angular degrees (lon, lat). Naive Euclidean distance in degrees causes severe distortion.
   - *Mitigation*: We will implement projection to local UTM metric coordinates or exact Haversine / geodesic distance calculations in `src/pathsnap_loader.py`.
2. **Ground Truth Association**: The reference `.route` is a sequence of traversed road segments (`arc_id`s), not a point-by-point label for each row in `.track`.
   - *Mitigation*: We will establish a rigorous projection and time-aligned spatial association to assign ground-truth labels to candidate arcs without leakage.
3. **CPU Execution Speed**: 80 trips with ~1000 GPS points each could generate millions of candidate-pair rows.
   - *Mitigation*: Spatial index (KDTree / BallTree / R-tree) with constrained radius (e.g. 30m-50m) to generate at most 5-15 candidates per GPS point, keeping memory and CPU footprint small.
