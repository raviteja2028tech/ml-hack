"""
Phase 7: PathSnap FastAPI backend.

Endpoints
    GET  /health    service status, whether the LightGBM model is loaded
    GET  /trips     available student-test trip ids
    POST /predict   {"trip_id": "00000052"} -> real end-to-end inference (LightGBM + HMM/Viterbi)
    GET  /metrics   saved five-model comparison + HMM evaluation summary
    GET  /ui/       the evaluator dashboard (src/phase8_ui), served for single-laptop demos

Run (LAN):  .venv\\Scripts\\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
Env vars:   PATHSNAP_CORS_ORIGINS  comma-separated origins (default "*")
            PATHSNAP_MODEL         model bundle path (default artifacts/models/lightgbm.joblib)
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.phase6_pipeline.end_to_end_matcher import DEFAULT_MODEL, PathSnapPipeline, list_trips

ROOT = Path(__file__).resolve().parents[2]
UI_DIR = ROOT / "src" / "phase8_ui"
METRICS_DIR = ROOT / "results" / "metrics"
PARTITION = "student_test"
CACHE_SIZE = 16


class PredictRequest(BaseModel):
    trip_id: str = Field(..., pattern=r"^\d{8}$", examples=["00000052"])
    include_geometry: bool = True


class _State:
    def __init__(self):
        self.pipeline: Optional[PathSnapPipeline] = None
        self.error: Optional[str] = None
        self.load_ms: Optional[float] = None
        self.lock = threading.Lock()
        self.cache: "OrderedDict[tuple, Dict]" = OrderedDict()
        self.trip_sizes: Dict[str, int] = {}


STATE = _State()


def _safe(msg: str) -> str:
    """Never leak absolute local paths in error messages."""
    return str(msg).replace(str(ROOT), "<project>").replace(str(ROOT).replace("\\", "/"), "<project>")


def get_pipeline() -> PathSnapPipeline:
    with STATE.lock:
        if STATE.pipeline is None:
            t = time.perf_counter()
            try:
                STATE.pipeline = PathSnapPipeline(Path(os.environ.get("PATHSNAP_MODEL", DEFAULT_MODEL)))
                STATE.error = None
                STATE.load_ms = (time.perf_counter() - t) * 1e3
            except Exception as e:  # noqa: BLE001
                STATE.error = _safe(f"{type(e).__name__}: {e}")
                raise HTTPException(status_code=503, detail=f"model not available: {STATE.error}")
        return STATE.pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_pipeline()  # load the model once at startup
        print(f"[PathSnap] model loaded in {STATE.load_ms:.0f} ms; {len(list_trips(PARTITION))} student-test trips", flush=True)
    except HTTPException:
        print(f"[PathSnap] WARNING model failed to load: {STATE.error}", flush=True)
    yield


app = FastAPI(
    title="PathSnap Map-Matching API",
    version="2.0.0",
    description="GPS map matching: candidate generation + LightGBM emissions + HMM/Viterbi decoding",
    lifespan=lifespan,
)
_origins = [o.strip() for o in os.environ.get("PATHSNAP_CORS_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["*"])


@app.get("/health")
def health() -> Dict:
    loaded = STATE.pipeline is not None
    return {
        "status": "ok" if loaded else "degraded",
        "service": "PathSnap API",
        "version": app.version,
        "model_loaded": loaded,
        "model": STATE.pipeline.model_name if loaded else None,
        "model_file": Path(STATE.pipeline.model_file).name if loaded else None,
        "decoder": "HMM + Viterbi",
        "trips_available": len(list_trips(PARTITION)),
        "error": STATE.error,
    }


@app.get("/trips")
def trips() -> Dict:
    ids = list_trips(PARTITION)
    out = []
    for tid in ids:
        if tid not in STATE.trip_sizes:
            p = ROOT / "data" / "raw" / "Student folder" / PARTITION / tid / f"{tid}.track"
            with open(p, "rb") as f:
                STATE.trip_sizes[tid] = sum(1 for line in f if line.strip())
        out.append({"trip_id": tid, "n_points": STATE.trip_sizes[tid]})
    return {"partition": PARTITION, "count": len(out), "trips": out}


@app.post("/predict")
def predict(req: PredictRequest) -> Dict:
    if req.trip_id not in list_trips(PARTITION):
        raise HTTPException(status_code=404, detail=f"unknown student-test trip {req.trip_id}")
    key = (req.trip_id, req.include_geometry)
    if key in STATE.cache:
        STATE.cache.move_to_end(key)
        return {**STATE.cache[key], "cached": True}
    pipe = get_pipeline()
    t = time.perf_counter()
    try:
        result = pipe.run(req.trip_id, partition=PARTITION, evaluate=True, include_geometry=req.include_geometry)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=_safe(str(e)))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=_safe(str(e)))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=_safe(f"{type(e).__name__}: {e}"))
    result["server_ms"] = round((time.perf_counter() - t) * 1e3, 1)
    result["error"] = None
    print(f"[PathSnap] /predict {req.trip_id}: {result['n_points']} pts, "
          f"inference {result['timing_ms']['inference_total']:.0f} ms", flush=True)
    STATE.cache[key] = result
    while len(STATE.cache) > CACHE_SIZE:
        STATE.cache.popitem(last=False)
    return {**result, "cached": False}


def _read_json(name: str) -> Optional[Dict]:
    p = METRICS_DIR / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/metrics")
def metrics() -> Dict:
    cmp_ = _read_json("model_comparison.json")
    hmm = _read_json("hmm_evaluation.json")
    tuning = _read_json("optuna_lightgbm.json")
    keep = ("model", "family", "cv_top1_mean", "cv_top1_std", "test_top1", "test_f1", "test_logloss",
            "test_roc_auc", "test_continuity_reach3", "final_fit_s", "model_file_mb")
    return {
        "model_comparison": None if cmp_ is None else {
            "selection_rule": cmp_.get("selection_rule"),
            "best_model": cmp_.get("best_model"),
            "models": [{k: r.get(k) for k in keep} for r in cmp_.get("comparison", [])],
            "heuristic_baselines_student_test": cmp_.get("heuristic_baselines_student_test"),
        },
        "hmm_evaluation": None if hmm is None else {
            "trips": hmm.get("trips"), "overall": hmm.get("overall"), "hmm_config": hmm.get("hmm_config"),
            "note": hmm.get("note"),
        },
        "bayesian_optimization": None if tuning is None else {
            k: tuning.get(k) for k in ("n_trials", "cv_protocol", "baseline_cv_top1", "best_cv_top1", "best_params", "decision")
        },
    }


if UI_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(UI_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root():
        return RedirectResponse(url="/ui/")
