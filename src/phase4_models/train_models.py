"""
PathSnap Phase 4 - five-model candidate-classifier benchmark.

    .venv\\Scripts\\python.exe src\\train_models.py --smoke        # tiny end-to-end check
    .venv\\Scripts\\python.exe src\\train_models.py                # full run (resumable)

Protocol
    * Data: Phase 3 candidates, labelled GPS points only (status ok/boundary).
    * CV: 5-fold GroupKFold by trip on a trip-balanced subset (at most CV_CAP labelled
      points per trip, all of each point's candidates kept). This keeps heavy trips
      (21/84: ~16k points) from dominating and keeps CV inside the RAM budget.
    * Final: refit on ALL labelled points of the 80 train trips, then evaluate on the
      10 student-test trips (development comparison).
    * Preprocessing (QuantileTransformer for LR/MLP) lives inside the sklearn Pipeline,
      so it is fitted on training folds only.
    * Primary metric: observation-level Top-1 accuracy (argmax probability per GPS point).
Each model's result and artifact are saved as soon as it finishes; reruns skip finished models.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.feature_engineering import FEATURE_COLUMNS  # noqa: E402
from src.phase3_dataset import load_candidates, load_points  # noqa: E402

RAW = ROOT / "data" / "raw" / "Student folder"
OUT_DIR = ROOT / "outputs" / "reports" / "phase4"
MODEL_DIR = ROOT / "models" / "phase4"
SEED = 42
CV_CAP = 1000
N_FOLDS = 5
N_JOBS = 6
MODEL_ORDER = ["logistic_regression", "random_forest", "extra_trees", "lightgbm", "mlp"]
ID_COLS = ["trip_id", "point_idx", "segment_id", "arc_id", "label"]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def make_model(name: str, smoke: bool = False):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import QuantileTransformer

    qt = lambda: QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000, random_state=SEED)  # noqa: E731
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return make_pipeline(qt(), LogisticRegression(C=1.0, max_iter=50 if smoke else 500, random_state=SEED))
    if name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=10 if smoke else 100, min_samples_leaf=50, max_features="sqrt", max_samples=0.3,
            n_jobs=N_JOBS, random_state=SEED,
        )
    if name == "extra_trees":
        from sklearn.ensemble import ExtraTreesClassifier

        return ExtraTreesClassifier(
            n_estimators=10 if smoke else 100, min_samples_leaf=50, max_features="sqrt", bootstrap=True, max_samples=0.3,
            n_jobs=N_JOBS, random_state=SEED,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=20 if smoke else 500, learning_rate=0.05, num_leaves=63, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            n_jobs=N_JOBS, random_state=SEED, verbose=-1,
        )
    if name == "mlp":
        from sklearn.neural_network import MLPClassifier

        return make_pipeline(
            qt(),
            MLPClassifier(
                hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=1024, learning_rate_init=1e-3,
                max_iter=2 if smoke else 30, early_stopping=True, n_iter_no_change=4, random_state=SEED,
            ),
        )
    raise ValueError(name)


MODEL_FAMILY = {
    "logistic_regression": "Linear / probabilistic",
    "random_forest": "Ensemble (bagging)",
    "extra_trees": "Randomised trees ensemble",
    "lightgbm": "Gradient boosting (histogram)",
    "mlp": "Neural network",
}


def model_config(model) -> dict:
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    keep = (int, float, str, bool, type(None), tuple)
    return {k: v for k, v in est.get_params().items() if isinstance(v, keep)}


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def point_keys(df: pd.DataFrame) -> np.ndarray:
    codes = pd.Categorical(df["trip_id"]).codes.astype(np.int64)
    return codes * 10_000_000 + df["point_idx"].to_numpy(np.int64)


def top1_rows(df: pd.DataFrame, proba: np.ndarray) -> np.ndarray:
    """Row index (positional) of the highest-probability candidate of each GPS point."""
    key = point_keys(df)
    order = np.lexsort((-proba, key))
    first = np.r_[True, key[order][1:] != key[order][:-1]]
    return order[first]


class ArcTopology:
    """Successor lists per trip from the raw .arcs files (cheap; no geometry needed)."""

    def __init__(self):
        self._cache = {}

    def get(self, trip_id: str):
        if trip_id not in self._cache:
            part = "train" if (RAW / "train" / trip_id).exists() else "student_test"
            arcs = pd.read_csv(RAW / part / trip_id / f"{trip_id}.arcs", sep=r"\s+", header=None, dtype=np.int64).to_numpy()
            fr, to = arcs[:, 0], arcs[:, 1]
            order = np.argsort(fr, kind="stable")
            ptr = np.r_[0, np.cumsum(np.bincount(fr, minlength=max(fr.max(), to.max()) + 1))]
            self._cache[trip_id] = (fr, to, order, ptr)
        return self._cache[trip_id]

    def hops_le(self, trip_id: str, a: int, b: int, max_hops: int = 3) -> bool:
        fr, to, order, ptr = self.get(trip_id)
        if a == b:
            return True
        frontier = {a}
        seen = {a}
        for _ in range(max_hops):
            nxt = set()
            for x in frontier:
                n = to[x]
                for y in order[ptr[n]:ptr[n + 1]].tolist():
                    if y == b:
                        return True
                    if y not in seen:
                        seen.add(y)
                        nxt.add(y)
            frontier = nxt
            if not frontier:
                break
        return False


def route_continuity(pred: pd.DataFrame, topo: ArcTopology) -> dict:
    """
    pred: one row per GPS point (trip_id, point_idx, segment_id, arc_id) in any order.
    direct  = consecutive points share the arc or arc_i.to_node == arc_{i+1}.from_node
    reach3  = arc_{i+1} reachable from arc_i within <= 3 hops (allows short arcs skipped between 1 Hz samples)
    Only consecutive points in the same time segment are compared.
    """
    pred = pred.sort_values(["trip_id", "point_idx"])
    n_pairs = n_direct = n_reach = 0
    for tid, g in pred.groupby("trip_id", sort=False):
        fr, to, _, _ = topo.get(tid)
        a = g["arc_id"].to_numpy(np.int64)
        s = g["segment_id"].to_numpy()
        p = g["point_idx"].to_numpy()
        ok = (s[1:] == s[:-1]) & (np.diff(p) == 1)
        x, y = a[:-1][ok], a[1:][ok]
        direct = (x == y) | (to[x] == fr[y])
        n_pairs += ok.sum()
        n_direct += direct.sum()
        cache = {}
        reach = direct.copy()
        for i in np.flatnonzero(~direct):
            k = (int(x[i]), int(y[i]))
            if k not in cache:
                cache[k] = topo.hops_le(tid, *k)
            reach[i] = cache[k]
        n_reach += reach.sum()
    return {
        "pairs": int(n_pairs),
        "continuity_direct": float(n_direct / max(n_pairs, 1)),
        "continuity_reach3": float(n_reach / max(n_pairs, 1)),
    }


def evaluate(df: pd.DataFrame, proba: np.ndarray, topo: ArcTopology, n_points_total: int | None = None) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score

    y = df["label"].to_numpy()
    yhat = (proba >= 0.5).astype(np.int8)
    rows = top1_rows(df, proba)
    top = df.iloc[rows]
    n_pts = len(rows)
    m = {
        "rows": int(len(df)),
        "points": int(n_pts),
        "top1_accuracy": float(top["label"].mean()),
        "accuracy_rows": float(accuracy_score(y, yhat)),
        "precision": float(precision_score(y, yhat, zero_division=0)),
        "recall": float(recall_score(y, yhat, zero_division=0)),
        "f1": float(f1_score(y, yhat, zero_division=0)),
        "f1_macro": float(f1_score(y, yhat, average="macro", zero_division=0)),
        "log_loss": float(log_loss(y, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y, proba)),
        "confusion_matrix": confusion_matrix(y, yhat, labels=[0, 1]).tolist(),
    }
    if n_points_total:
        m["top1_accuracy_all_points"] = float(top["label"].sum() / n_points_total)
    m.update({f"pred_{k}": v for k, v in route_continuity(top[["trip_id", "point_idx", "segment_id", "arc_id"]], topo).items()})
    return m


def fold_summary(folds: list) -> dict:
    keys = [k for k, v in folds[0].items() if isinstance(v, float)]
    return {k: {"mean": float(np.mean([f[k] for f in folds])), "std": float(np.std([f[k] for f in folds]))} for k in keys}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_split(partition: str, trips=None) -> pd.DataFrame:
    df = load_candidates(partition, columns=ID_COLS + FEATURE_COLUMNS, labelled_only=True, trips=trips)
    df["trip_id"] = df["trip_id"].astype(str)
    return df.reset_index(drop=True)


def trip_balanced_subset(df: pd.DataFrame, cap: int, seed: int = SEED) -> pd.DataFrame:
    pts = df[["trip_id", "point_idx"]].drop_duplicates()
    rng = np.random.default_rng(seed)
    pts = pts.assign(r=rng.random(len(pts))).sort_values(["trip_id", "r"])
    pts = pts[pts.groupby("trip_id").cumcount() < cap][["trip_id", "point_idx"]]
    return df.merge(pts, on=["trip_id", "point_idx"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def run(models, smoke: bool, force: bool, cv_cap: int, n_folds: int):
    import joblib
    import psutil
    from sklearn.model_selection import GroupKFold

    out_dir = OUT_DIR / ("smoke" if smoke else "")
    model_dir = MODEL_DIR / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    proc = psutil.Process()
    topo = ArcTopology()

    t0 = time.perf_counter()
    train_trips = None
    test_trips = None
    if smoke:
        all_train = sorted(p.name for p in (RAW / "train").iterdir())
        train_trips = all_train[:6]
        test_trips = ["00000052", "00000065"]
    train = load_split("train", train_trips)
    test = load_split("student_test", test_trips)
    test_points_total = len(load_points("student_test", test_trips))
    cv_df = trip_balanced_subset(train, 200 if smoke else cv_cap)
    print(f"[data] train rows={len(train):,} test rows={len(test):,} cv rows={len(cv_df):,} "
          f"cv points={cv_df[['trip_id','point_idx']].drop_duplicates().shape[0]:,} load={time.perf_counter()-t0:.1f}s "
          f"rss={proc.memory_info().rss/1e6:.0f}MB", flush=True)
    assert not set(train.trip_id) & set(test.trip_id), "train/test trip overlap"

    folds_idx = list(GroupKFold(n_splits=2 if smoke else n_folds).split(cv_df, groups=cv_df["trip_id"]))
    for tr_i, va_i in folds_idx:
        assert not set(cv_df.trip_id.iloc[tr_i]) & set(cv_df.trip_id.iloc[va_i])

    X_cv = cv_df[FEATURE_COLUMNS].to_numpy(np.float32)
    y_cv = cv_df["label"].to_numpy()

    for name in models:
        res_path = out_dir / f"{name}.json"
        if res_path.exists() and not force:
            print(f"[skip] {name} (result exists)", flush=True)
            continue
        print(f"[model] {name}", flush=True)
        tm = time.perf_counter()
        folds = []
        for k, (tr_i, va_i) in enumerate(folds_idx):
            mdl = make_model(name, smoke)
            ts = time.perf_counter()
            mdl.fit(X_cv[tr_i], y_cv[tr_i])
            fit_s = time.perf_counter() - ts
            ts = time.perf_counter()
            p = mdl.predict_proba(X_cv[va_i])[:, 1]
            pred_s = time.perf_counter() - ts
            m = evaluate(cv_df.iloc[va_i], p, topo)
            m.update(fit_s=fit_s, predict_s=pred_s, val_trips=int(cv_df.trip_id.iloc[va_i].nunique()))
            folds.append(m)
            print(f"   fold {k}: top1={m['top1_accuracy']:.4f} f1={m['f1']:.4f} logloss={m['log_loss']:.4f} "
                  f"fit={fit_s:.1f}s rss={proc.memory_info().rss/1e6:.0f}MB", flush=True)
            del mdl
            gc.collect()

        # final fit on all labelled train points -> student-test
        mdl = make_model(name, smoke)
        X_tr = train[FEATURE_COLUMNS].to_numpy(np.float32)
        ts = time.perf_counter()
        mdl.fit(X_tr, train["label"].to_numpy())
        final_fit_s = time.perf_counter() - ts
        del X_tr
        gc.collect()
        X_te = test[FEATURE_COLUMNS].to_numpy(np.float32)
        ts = time.perf_counter()
        p_te = mdl.predict_proba(X_te)[:, 1]
        infer_s = time.perf_counter() - ts
        test_m = evaluate(test, p_te, topo, n_points_total=test_points_total)
        test_m.update(
            inference_s=infer_s,
            inference_ms_per_1k_rows=1000 * infer_s / max(len(test), 1) * 1000,
            inference_s_per_trip=infer_s / test.trip_id.nunique(),
        )
        model_path = model_dir / f"{name}.joblib"
        joblib.dump({"model": mdl, "features": FEATURE_COLUMNS, "name": name}, model_path, compress=3)
        result = {
            "model": name,
            "family": MODEL_FAMILY[name],
            "config": model_config(mdl),
            "preprocessing": "QuantileTransformer(normal) in-pipeline" if hasattr(mdl, "steps") else "none (trees)",
            "cv": {"scheme": f"GroupKFold({len(folds_idx)}) by trip, trip-balanced cap={200 if smoke else cv_cap} pts/trip",
                   "summary": fold_summary(folds), "folds": folds},
            "final_fit_s": final_fit_s,
            "train_rows_final": int(len(train)),
            "student_test": test_m,
            "model_file": str(model_path.relative_to(ROOT)),
            "model_file_mb": round(model_path.stat().st_size / 1e6, 2),
            "wall_s": time.perf_counter() - tm,
            "peak_rss_mb_so_far": round(getattr(proc.memory_info(), "peak_wset", 0) / 1e6, 1),
        }
        res_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        s = result["cv"]["summary"]
        print(f"   => CV top1={s['top1_accuracy']['mean']:.4f}±{s['top1_accuracy']['std']:.4f} | "
              f"student-test top1={test_m['top1_accuracy']:.4f} f1={test_m['f1']:.4f} logloss={test_m['log_loss']:.4f} "
              f"cont(reach3)={test_m['pred_continuity_reach3']:.4f} | final fit {final_fit_s:.1f}s | wall {result['wall_s']:.1f}s", flush=True)
        del mdl, p_te, X_te
        gc.collect()

    compare(out_dir, test, topo, test_points_total, time.perf_counter() - t0)


def compare(out_dir: Path, test: pd.DataFrame, topo: ArcTopology, test_points_total: int, elapsed: float):
    rows = []
    for name in MODEL_ORDER:
        f = out_dir / f"{name}.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text(encoding="utf-8"))
        s, t = r["cv"]["summary"], r["student_test"]
        rows.append({
            "model": name, "family": r["family"],
            "cv_top1_mean": s["top1_accuracy"]["mean"], "cv_top1_std": s["top1_accuracy"]["std"],
            "cv_f1_mean": s["f1"]["mean"], "cv_logloss_mean": s["log_loss"]["mean"],
            "test_top1": t["top1_accuracy"], "test_top1_all_points": t.get("top1_accuracy_all_points"),
            "test_precision": t["precision"], "test_recall": t["recall"], "test_f1": t["f1"], "test_f1_macro": t["f1_macro"],
            "test_logloss": t["log_loss"], "test_roc_auc": t["roc_auc"],
            "test_continuity_direct": t["pred_continuity_direct"], "test_continuity_reach3": t["pred_continuity_reach3"],
            "cv_fit_s_mean": s["fit_s"]["mean"], "final_fit_s": r["final_fit_s"],
            "test_inference_s_per_trip": t["inference_s_per_trip"], "model_file_mb": r["model_file_mb"],
        })
    if not rows:
        return
    table = pd.DataFrame(rows)
    # selection: highest mean CV Top-1 (generalisation across trips); student-test reported, not used to tune
    best = table.sort_values(["cv_top1_mean", "cv_logloss_mean"], ascending=[False, True]).iloc[0]

    # reference rows: label-free heuristics and the reference labels themselves on student-test
    ref = {}
    score_near = -test["dist_m"].to_numpy(np.float64)
    score_anchor = -(test["dist_m"].to_numpy(np.float64) + 7.5 * (1 - test["heading_cos"].to_numpy(np.float64)))
    for nm, sc in [("nearest_distance", score_near), ("distance_plus_heading", score_anchor)]:
        top = test.iloc[top1_rows(test, sc)]
        ref[nm] = {"test_top1": float(top["label"].mean()),
                   **{f"test_{k}": v for k, v in route_continuity(top[["trip_id", "point_idx", "segment_id", "arc_id"]], topo).items()}}
    truth = test[test["label"] == 1]
    ref["reference_labels"] = {f"test_{k}": v for k, v in route_continuity(truth[["trip_id", "point_idx", "segment_id", "arc_id"]], topo).items()}

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "selection_rule": "max mean 5-fold GroupKFold Top-1 accuracy (tie-break: lower CV log loss); student-test not used for selection",
        "best_model": best["model"],
        "best_cv_top1": float(best["cv_top1_mean"]),
        "best_test_top1": float(best["test_top1"]),
        "comparison": rows,
        "heuristic_baselines_student_test": ref,
        "elapsed_this_run_s": elapsed,
    }
    (out_dir / "model_comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    cols = ["model", "cv_top1_mean", "cv_top1_std", "test_top1", "test_f1", "test_logloss", "test_continuity_reach3", "final_fit_s"]
    print("\n" + table[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBEST: {best['model']}  CV top1={best['cv_top1_mean']:.4f}  student-test top1={best['test_top1']:.4f}")
    print("heuristics (student-test): " + json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items() if isinstance(vv, float)} for k, v in ref.items()}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--models", nargs="+", default=MODEL_ORDER)
    ap.add_argument("--cv-cap", type=int, default=CV_CAP)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    a = ap.parse_args()
    run(a.models, a.smoke, a.force, a.cv_cap, a.folds)


if __name__ == "__main__":
    main()
