"""
PathSnap core package.

Modules live in phase-wise subpackages. Legacy import paths such as
``from src.candidate_generator import ...`` keep working through the aliases
below. Each alias is registered right after its module loads, in dependency
order, so every module is loaded exactly once (no duplicate module objects).
"""

import importlib
import sys

_LEGACY_ALIASES = (  # dependency order matters
    ("pathsnap_loader", "src.phase1_eda.pathsnap_loader"),
    ("visualize_folium", "src.phase1_eda.visualize_folium"),
    ("candidate_generator", "src.phase3_features.candidate_generator"),
    ("feature_engineering", "src.phase3_features.feature_engineering"),
    ("route_labeler", "src.phase3_features.route_labeler"),
    ("phase3_dataset", "src.phase3_features.phase3_dataset"),
    ("phase3_pipeline", "src.phase3_features.phase3_pipeline"),
    ("train_models", "src.phase4_models.train_models"),
)

for _name, _target in _LEGACY_ALIASES:
    _module = importlib.import_module(_target)
    sys.modules[f"src.{_name}"] = _module
    globals()[_name] = _module
del _name, _target, _module
