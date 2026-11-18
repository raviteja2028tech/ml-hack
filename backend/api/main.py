"""
Backward-compatible entry point for the PathSnap FastAPI backend (implementation: src/phase7_api/app.py).

    .venv\\Scripts\\python.exe -m uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from src.phase7_api.app import app  # noqa: F401
