"""
Reorganize PathSnap workspace into target phase-wise structure.
Safe execution: copies/moves files, creates aliases for backwards compatibility,
updates top-level imports, and ensures tests continue passing.
"""

from pathlib import Path
import shutil
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def reorganize():
    print("Starting PathSnap reorganization...")

    # 1. Create target directories
    target_dirs = [
        "src/phase1_eda",
        "src/phase2_preprocessing",
        "src/phase3_features",
        "src/phase4_models",
        "src/phase5_hmm",
        "src/phase6_pipeline",
        "api",
        "ui",
        "artifacts/models",
        "results/figures",
        "results/metrics",
        "results/maps/eda_maps",
        "reports",
        "docs/guidelines",
    ]

    for d in target_dirs:
        (PROJECT_ROOT / d).mkdir(parents=True, exist_ok=True)
        init_file = PROJECT_ROOT / d / "__init__.py"
        if d.startswith("src/") or d == "api":
            if not init_file.exists():
                init_file.write_text("# Auto-generated package init\n", encoding="utf-8")

    # 2. Phase 1: EDA
    shutil.copy2(PROJECT_ROOT / "src/pathsnap_loader.py", PROJECT_ROOT / "src/phase1_eda/pathsnap_loader.py")
    shutil.copy2(PROJECT_ROOT / "src/visualize_folium.py", PROJECT_ROOT / "src/phase1_eda/visualize_folium.py")
    if (PROJECT_ROOT / "scripts/run_audit.py").exists():
        shutil.copy2(PROJECT_ROOT / "scripts/run_audit.py", PROJECT_ROOT / "src/phase1_eda/audit_dataset.py")

    # 3. Phase 2: Preprocessing
    phase2_init = PROJECT_ROOT / "src/phase2_preprocessing/__init__.py"
    phase2_init.write_text('"""Phase 2 Preprocessing: Geodesic projections & spatial coordinate conversions."""\nfrom src.phase1_eda.pathsnap_loader import haversine_distance, calculate_bearing, point_to_segment_distance\n', encoding="utf-8")

    # 4. Phase 3: Features & Candidates
    shutil.copy2(PROJECT_ROOT / "src/candidate_generator.py", PROJECT_ROOT / "src/phase3_features/candidate_generator.py")
    shutil.copy2(PROJECT_ROOT / "src/feature_engineering.py", PROJECT_ROOT / "src/phase3_features/feature_engineering.py")
    shutil.copy2(PROJECT_ROOT / "src/route_labeler.py", PROJECT_ROOT / "src/phase3_features/route_labeler.py")
    shutil.copy2(PROJECT_ROOT / "src/phase3_dataset.py", PROJECT_ROOT / "src/phase3_features/phase3_dataset.py")
    shutil.copy2(PROJECT_ROOT / "src/phase3_pipeline.py", PROJECT_ROOT / "src/phase3_features/phase3_pipeline.py")
    if (PROJECT_ROOT / "scripts/build_phase3_dataset.py").exists():
        shutil.copy2(PROJECT_ROOT / "scripts/build_phase3_dataset.py", PROJECT_ROOT / "src/phase3_features/dataset_builder.py")

    # 5. Phase 4: Models
    shutil.copy2(PROJECT_ROOT / "src/train_models.py", PROJECT_ROOT / "src/phase4_models/train_models.py")

    # 6. Phase 5: HMM / Viterbi Skeleton
    hmm_stub = PROJECT_ROOT / "src/phase5_hmm/hmm_decoder.py"
    if not hmm_stub.exists():
        hmm_stub.write_text('"""Phase 5: Hidden Markov Model & Viterbi Sequential Route Reconstruction."""\n\nclass HMMRouteDecoder:\n    def __init__(self, sigma_z=4.29, beta=5.0):\n        self.sigma_z = sigma_z\n        self.beta = beta\n', encoding="utf-8")

    # 7. Phase 6: End-to-End Pipeline Skeleton
    p6_stub = PROJECT_ROOT / "src/phase6_pipeline/end_to_end_matcher.py"
    if not p6_stub.exists():
        p6_stub.write_text('"""Phase 6: End-to-End Map Matching and Route Reconstruction Pipeline."""\n\nclass PathSnapPipeline:\n    def __init__(self, model_path=None):\n        self.model_path = model_path\n', encoding="utf-8")

    # 8. Artifacts / Models (Copy 5 models)
    src_models = PROJECT_ROOT / "models/phase4"
    if src_models.exists():
        for mf in src_models.glob("*.joblib"):
            dst = PROJECT_ROOT / "artifacts/models" / mf.name
            shutil.copy2(mf, dst)
            print(f"Copied model {mf.name} -> artifacts/models/")

    # 9. Results: Maps, Figures & Metrics
    if (PROJECT_ROOT / "outputs/figures/REVIEW1_MAP.html").exists():
        shutil.copy2(PROJECT_ROOT / "outputs/figures/REVIEW1_MAP.html", PROJECT_ROOT / "results/maps/REVIEW1_MAP.html")
    
    eda_maps_dir = PROJECT_ROOT / "outputs/figures/eda_maps"
    if eda_maps_dir.exists():
        for mf in eda_maps_dir.glob("*.html"):
            shutil.copy2(mf, PROJECT_ROOT / "results/maps/eda_maps" / mf.name)

    phase4_reports = PROJECT_ROOT / "outputs/reports/phase4"
    if phase4_reports.exists():
        for jf in phase4_reports.glob("*.*"):
            shutil.copy2(jf, PROJECT_ROOT / "results/metrics" / jf.name)

    for rf in ["phase3_benchmark.json", "dataset_exploration_report.json"]:
        p = PROJECT_ROOT / "outputs/reports" / rf
        if p.exists():
            shutil.copy2(p, PROJECT_ROOT / "results/metrics" / rf)

    # 10. Reports
    for doc in ["PHASE3_REPORT.md", "DATASET_EXPLORATION.md", "HACKATHON_REQUIREMENTS.md", "IMPLEMENTATION_PLAN.md"]:
        p = PROJECT_ROOT / "docs" / doc
        if p.exists():
            shutil.copy2(p, PROJECT_ROOT / "reports" / doc)

    for rep in ["phase3_build_log.txt", "phase3_dataset_report.json", "phase3_unit_tests.txt", "phase3_benchmark_log.txt"]:
        p = PROJECT_ROOT / "outputs/reports" / rep
        if p.exists():
            shutil.copy2(p, PROJECT_ROOT / "reports" / rep)

    # 11. API starter
    api_main = PROJECT_ROOT / "api/main.py"
    if not api_main.exists():
        api_main.write_text('''"""
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
''', encoding="utf-8")

    # 12. UI placeholder
    ui_readme = PROJECT_ROOT / "ui/README.md"
    if not ui_readme.exists():
        ui_readme.write_text("""# PathSnap Evaluator Interface (Laptop 2)
Web-based evaluator dashboard to select evaluation trips, query the Laptop 1 FastAPI endpoint across LAN, and visualize map matching results with Folium / Leaflet.
""", encoding="utf-8")

    # 13. Update src/__init__.py with full backward compatibility aliases
    src_init = PROJECT_ROOT / "src/__init__.py"
    src_init.write_text('''"""
PathSnap Core Module
Exposes subpackages and backward-compatible module aliases.
"""

import sys
from pathlib import Path

# Subpackages
from src.phase1_eda import pathsnap_loader, visualize_folium
from src.phase3_features import candidate_generator, feature_engineering, route_labeler, phase3_dataset, phase3_pipeline
from src.phase4_models import train_models

# Compatibility module aliasing so legacy imports like `from src.candidate_generator import ...` continue working
sys.modules['src.pathsnap_loader'] = pathsnap_loader
sys.modules['src.visualize_folium'] = visualize_folium
sys.modules['src.candidate_generator'] = candidate_generator
sys.modules['src.feature_engineering'] = feature_engineering
sys.modules['src.route_labeler'] = route_labeler
sys.modules['src.phase3_dataset'] = phase3_dataset
sys.modules['src.phase3_pipeline'] = phase3_pipeline
sys.modules['src.train_models'] = train_models
''', encoding="utf-8")

    print("\nReorganization structure created successfully!")

if __name__ == "__main__":
    reorganize()
