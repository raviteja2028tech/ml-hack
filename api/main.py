"""
FastAPI Backend Service for PathSnap (Laptop 1 Inference Server).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

app = FastAPI(
    title="PathSnap Map-Matching API",
    version="1.0.0",
    description="Distributed Map-Matching and Route Reconstruction Engine"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PathSnap API", "version": "1.0.0"}
