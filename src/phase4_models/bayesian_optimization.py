"""
Optuna (TPE) Bayesian optimisation of the LightGBM candidate classifier.

* Training data only: 80 train trips, trip-balanced subset, GroupKFold by trip.
  Student-test is never loaded here, so it cannot influence the selection.
* Trial 0 is the deployed Phase 4 configuration, scored with the SAME protocol, so
  "tuned vs baseline" is an apples-to-apples comparison.
* The deployed model (artifacts/models/lightgbm.joblib) is never modified. Results go to
  results/metrics/optuna_lightgbm.json (+ per-trial CSV).

    python -m src.phase4_models.bayesian_optimization --trials 12 --timeout 600
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.phase3_features.feature_engineering import FEATURE_COLUMNS
from src.phase4_models.train_models import SEED, load_split, top1_rows, trip_balanced_subset

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "metrics" / "optuna_lightgbm.json"

BASELINE = {  # Phase 4 deployed LightGBM
    "n_estimators": 500, "learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 100,
    "subsample": 0.8, "colsample_bytree": 0.8, "reg_lambda": 1.0,
}


def cv_top1(params, X, y, df, folds, n_jobs):
    from lightgbm import LGBMClassifier

    scores = []
    for tr, va in folds:
        m = LGBMClassifier(**params, subsample_freq=1, n_jobs=n_jobs, random_state=SEED, verbose=-1)
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        sub = df.iloc[va]
        scores.append(float(sub["label"].to_numpy()[top1_rows(sub, p)].mean()))
    return float(np.mean(scores)), scores


def main(argv=None):
    import optuna
    from sklearn.model_selection import GroupKFold

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=12)
    ap.add_argument("--timeout", type=int, default=600, help="seconds")
    ap.add_argument("--cap", type=int, default=300, help="labelled points per trip in the tuning subset")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--n-jobs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    df = trip_balanced_subset(load_split("train"), cap=args.cap).reset_index(drop=True)
    X = df[FEATURE_COLUMNS].to_numpy(np.float32)
    y = df["label"].to_numpy()
    folds = list(GroupKFold(n_splits=args.folds).split(X, y, groups=df["trip_id"]))
    print(f"tuning subset: {df.trip_id.nunique()} trips, {len(df)} rows, {args.folds} grouped folds", flush=True)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 400, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        mean, scores = cv_top1(params, X, y, df, folds, args.n_jobs)
        trial.set_user_attr("fold_top1", scores)
        print(f"trial {trial.number}: CV Top-1 {mean:.4f}  {params}", flush=True)
        return mean

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED), study_name="lightgbm_top1")
    study.enqueue_trial(BASELINE)  # trial 0 = deployed configuration, same protocol
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    base = study.trials[0].value
    best = study.best_trial
    gain = best.value - base
    decision = (
        f"best trial {best.number} beats the deployed configuration by {gain * 100:.2f} pts CV Top-1 on the tuning subset; "
        "deployed model NOT replaced (needs a full retrain + grouped-CV confirmation first)"
        if best.number != 0 and gain > 0 else
        "no trial beat the deployed configuration; deployed model kept"
    )
    result = {
        "n_trials": len(study.trials),
        "cv_protocol": f"GroupKFold({args.folds}) by trip on {df.trip_id.nunique()} train trips, cap {args.cap} labelled points/trip "
                       f"({len(df)} candidate rows); metric = per-point Top-1; student-test not used",
        "baseline_params": BASELINE,
        "baseline_cv_top1": base,
        "best_cv_top1": best.value,
        "best_trial": best.number,
        "best_params": best.params,
        "improvement_pts": gain * 100,
        "decision": decision,
        "deployed_model_modified": False,
        "trials": [{"number": t.number, "value": t.value, "params": t.params, "fold_top1": t.user_attrs.get("fold_top1")}
                   for t in study.trials],
        "wall_s": time.perf_counter() - t0,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    study.trials_dataframe().to_csv(args.out.with_suffix(".trials.csv"), index=False)
    print(json.dumps({k: result[k] for k in ("baseline_cv_top1", "best_cv_top1", "best_trial", "improvement_pts", "decision")}, indent=2))


if __name__ == "__main__":
    main()
