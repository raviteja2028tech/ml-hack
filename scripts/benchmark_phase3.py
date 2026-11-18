"""
Phase 3 runtime / memory / accuracy benchmark.

    .venv\\Scripts\\python.exe scripts\\benchmark_phase3.py

A. Per-stage runtime (median of 3) and isolated peak memory for representative trips.
   Each run is in a fresh process. The label sensitivity re-runs are excluded (diagnostic only).
B. Candidate generator vs brute force (speed + exactness) and vs the Phase-2
   midpoint cKDTree (recall), on full trips.
C. GPS-to-road distance accuracy: local equirectangular metres vs haversine.
D. Cost of loading the consolidated training table (for Phase 4 RAM planning).
-> outputs/reports/phase3_benchmark.json
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RAW = ROOT / "data" / "raw" / "Student folder"
PROCESSED = ROOT / "data" / "processed" / "phase3"
OUT = ROOT / "outputs" / "reports" / "phase3_benchmark.json"

# small / typical / largest network / longest track / longest route (slowest labels)
BENCH_TRIPS = ["00000026", "00000001", "00000013", "00000054", "00000092"]


def _peak_mb():
    import psutil

    mi = psutil.Process().memory_info()
    return getattr(mi, "peak_wset", mi.rss) / 1e6


def _baseline_job():
    import numpy, pandas, scipy.spatial  # noqa: F401
    from src.phase3_pipeline import process_trip  # noqa: F401

    return _peak_mb()


def _stage_job(trip_id: str, with_labels: bool):
    from src.phase3_pipeline import process_trip

    r = process_trip(trip_id, base_dir=RAW, with_labels=with_labels)
    return {"timings": r.stats["timings"], "peak_mb": _peak_mb(), "rows": r.stats["n_candidate_rows"], "n_points": r.stats["n_points"], "n_arcs": r.stats["n_arcs"]}


def bench_stages():
    out = {}
    with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as ex:
        base = ex.submit(_baseline_job).result()
    out["baseline_process_peak_mb_after_imports"] = round(base, 1)
    for tid in BENCH_TRIPS:
        runs_lab, runs_inf = [], []
        for _ in range(3):
            with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as ex:
                runs_lab.append(ex.submit(_stage_job, tid, True).result())
            with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as ex:
                runs_inf.append(ex.submit(_stage_job, tid, False).result())
        keys = runs_lab[0]["timings"].keys()
        out[tid] = {
            "n_points": runs_lab[0]["n_points"],
            "n_arcs": runs_lab[0]["n_arcs"],
            "candidate_rows": runs_lab[0]["rows"],
            "median_timings_s_with_labels": {k: round(statistics.median(r["timings"][k] for r in runs_lab), 3) for k in keys},
            "median_total_s_inference_only": round(statistics.median(r["timings"]["total_s"] for r in runs_inf), 3),
            "peak_mb_with_labels": round(max(r["peak_mb"] for r in runs_lab), 1),
            "peak_mb_inference_only": round(max(r["peak_mb"] for r in runs_inf), 1),
        }
        print(f"[A] {tid}: {out[tid]}", flush=True)
    return out


def bench_candidates():
    import numpy as np

    from src.candidate_generator import ArcGeometry, CandidateConfig, brute_force_candidates, generate_candidates
    from src.pathsnap_loader import load_trip

    res = {}
    for tid in ["00000001", "00000013"]:
        trip = load_trip(tid, base_dir=RAW)
        geom = ArcGeometry.from_trip(trip)
        px, py = trip.track_df.x_m.to_numpy(), trip.track_df.y_m.to_numpy()
        r = {}
        for radius in (30.0, 50.0):
            t = time.perf_counter()
            ours = generate_candidates(geom, px, py, CandidateConfig(radius_m=radius))
            t_ours = time.perf_counter() - t
            t = time.perf_counter()
            bf = brute_force_candidates(geom, px, py, radius)
            t_bf = time.perf_counter() - t
            s_ours = set(zip(ours.point_idx.tolist(), ours.arc_id.tolist()))
            s_bf = set(zip(bf.point_idx.tolist(), bf.arc_id.tolist()))
            # Phase-2 style index: cKDTree on arc midpoints, queried with the same radius
            t = time.perf_counter()
            hits = trip.arc_kdtree.query_ball_point(np.column_stack([px, py]), r=radius)
            s_mid = {(i, int(a)) for i, h in enumerate(hits) for a in h}
            s_mid = {pa for pa in s_mid if pa in s_bf}  # count only true hits (midpoint within R => arc within R)
            t_mid = time.perf_counter() - t
            r[f"r{int(radius)}"] = {
                "pairs_true": len(s_bf),
                "ours_s": round(t_ours, 3),
                "brute_force_s": round(t_bf, 3),
                "speedup_vs_brute_force": round(t_bf / t_ours, 1),
                "ours_missing": len(s_bf - s_ours),
                "ours_extra": len(s_ours - s_bf),
                "midpoint_kdtree_s": round(t_mid, 3),
                "midpoint_kdtree_recall": round(len(s_mid) / max(len(s_bf), 1), 4),
            }
        res[tid] = {"n_points": len(px), "n_arcs": geom.n_arcs, **r}
        print(f"[B] {tid}: {res[tid]}", flush=True)
    return res


def bench_projection():
    import numpy as np
    import pyarrow.parquet as pq

    from src.candidate_generator import ArcGeometry, point_segment_distance
    from src.pathsnap_loader import load_trip

    R = 6371000.0
    rel, absd = [], []
    trips = sorted(p.stem for p in (PROCESSED / "train" / "candidates").glob("*.parquet"))
    rng = np.random.default_rng(0)
    for tid in trips[::4]:  # 20 trips spread over the partition
        trip = load_trip(tid, base_dir=RAW, partition="train")
        geom = ArcGeometry.from_trip(trip)
        c = pq.read_table(PROCESSED / "train" / "candidates" / f"{tid}.parquet", columns=["point_idx", "arc_id", "dist_m"]).to_pandas()
        c = c.iloc[rng.choice(len(c), size=min(2000, len(c)), replace=False)]
        p = c.point_idx.to_numpy()
        a = c.arc_id.to_numpy()
        px, py = trip.track_df.x_m.to_numpy()[p], trip.track_df.y_m.to_numpy()[p]
        d, t, _ = point_segment_distance(px, py, geom.ux[a], geom.uy[a], geom.vx[a], geom.vy[a])
        qx = geom.ux[a] + t * (geom.vx[a] - geom.ux[a])
        qy = geom.uy[a] + t * (geom.vy[a] - geom.uy[a])
        qlon = trip.origin_lon + qx / trip.m_per_deg_lon
        qlat = trip.origin_lat + qy / trip.m_per_deg_lat
        lon = trip.track_df.lon.to_numpy()[p]
        lat = trip.track_df.lat.to_numpy()[p]
        p1, p2 = np.radians(lat), np.radians(qlat)
        h = 2 * R * np.arcsin(np.sqrt(np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(qlon - lon) / 2) ** 2))
        ok = h > 1.0
        rel.append(np.abs(d[ok] - h[ok]) / h[ok])
        absd.append(np.abs(d - h))
    rel = np.concatenate(rel)
    absd = np.concatenate(absd)
    out = {
        "n_pairs": int(absd.size),
        "rel_err_median": float(np.median(rel)),
        "rel_err_p99": float(np.percentile(rel, 99)),
        "rel_err_max": float(rel.max()),
        "abs_err_m_p99": float(np.percentile(absd, 99)),
        "abs_err_m_max": float(absd.max()),
    }
    print(f"[C] projection vs haversine: {out}", flush=True)
    return out


def _load_job(columns):
    import pyarrow.parquet as pq

    t = time.perf_counter()
    df = pq.read_table(PROCESSED / "train_candidates.parquet", columns=columns).to_pandas()
    el = time.perf_counter() - t
    return {"rows": len(df), "cols": df.shape[1], "load_s": round(el, 2), "df_mb": round(df.memory_usage(deep=True).sum() / 1e6, 1), "peak_mb": round(_peak_mb(), 1)}


def bench_load():
    from src.feature_engineering import FEATURE_COLUMNS

    out = {}
    for name, cols in [("all_columns", None), ("features_plus_label", FEATURE_COLUMNS + ["trip_id", "point_idx", "label"])]:
        with ProcessPoolExecutor(max_workers=1, max_tasks_per_child=1) as ex:
            out[name] = ex.submit(_load_job, cols).result()
        print(f"[D] load {name}: {out[name]}", flush=True)
    return out


def main():
    rep = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    import platform

    import psutil

    rep["machine"] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpus": psutil.cpu_count(),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "ram_available_gb_at_start": round(psutil.virtual_memory().available / 1e9, 1),
        "python": platform.python_version(),
    }
    rep["A_stage_runtime_memory"] = bench_stages()
    rep["B_candidate_generator"] = bench_candidates()
    rep["C_projection_accuracy"] = bench_projection()
    if (PROCESSED / "train_candidates.parquet").exists():
        rep["D_dataset_load"] = bench_load()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
