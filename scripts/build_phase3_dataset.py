"""
Build the Phase 3 candidate / feature / label datasets.

    .venv\\Scripts\\python.exe scripts\\build_phase3_dataset.py                      # all 80 train + 10 student_test
    .venv\\Scripts\\python.exe scripts\\build_phase3_dataset.py --trips 00000001 00000075 --partitions train

Outputs (data/processed/phase3/):
    <partition>/candidates/<trip>.parquet   one row per (GPS point, candidate arc)
    <partition>/points/<trip>.parquet       one row per GPS point (label status, alignment)
    train_candidates.parquet, train_points.parquet                 (80 train trips, streamed)
    student_test_candidates.parquet, student_test_points.parquet   (10 trips, kept separate)
Report:  outputs/reports/phase3_dataset_report.json

Each trip runs in its own worker process (max_tasks_per_child=1), so the peak
memory of every trip is measured in isolation and released afterwards.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

RAW = ROOT / "data" / "raw" / "Student folder"
OUT = ROOT / "data" / "processed" / "phase3"
REPORT = ROOT / "outputs" / "reports" / "phase3_dataset_report.json"


def _worker(trip_id: str, partition: str, radius: float, out_dir: str, sensitivity) -> dict:
    import psutil
    import pyarrow as pa
    import pyarrow.parquet as pq

    from src.candidate_generator import CandidateConfig
    from src.phase3_pipeline import Phase3Config, process_trip

    cfg = Phase3Config(candidate=CandidateConfig(radius_m=radius))
    res = process_trip(trip_id, partition=partition, base_dir=RAW, config=cfg, label_sensitivity_weights=sensitivity)
    out = Path(out_dir) / partition
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    (out / "points").mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    pq.write_table(pa.Table.from_pandas(res.candidates, preserve_index=False), out / "candidates" / f"{trip_id}.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(res.points, preserve_index=False), out / "points" / f"{trip_id}.parquet", compression="zstd")
    res.stats["timings"]["write_s"] = time.perf_counter() - t
    mi = psutil.Process().memory_info()
    res.stats["peak_mem_mb"] = round(getattr(mi, "peak_wset", mi.rss) / 1e6, 1)
    res.stats["candidates_mem_mb"] = round(res.candidates.memory_usage(deep=True).sum() / 1e6, 2)
    return res.stats


def consolidate(partition: str, trips, out_dir: Path) -> dict:
    """Stream per-trip shards into one Parquet file (one row group per trip) - never loads all trips."""
    import pyarrow.parquet as pq

    info = {}
    for kind in ("candidates", "points"):
        dst = out_dir / f"{partition}_{kind}.parquet"
        writer = None
        rows = 0
        for tid in sorted(trips):
            tbl = pq.read_table(out_dir / partition / kind / f"{tid}.parquet")
            if writer is None:
                writer = pq.ParquetWriter(dst, tbl.schema, compression="zstd")
            writer.write_table(tbl)
            rows += tbl.num_rows
        if writer is not None:
            writer.close()
        info[kind] = {"file": str(dst.relative_to(ROOT)) if dst.is_relative_to(ROOT) else str(dst), "rows": rows, "size_mb": round(dst.stat().st_size / 1e6, 2)}
    return info


def radius_sweep(partition: str, trips, out_dir: Path, radii=(30.0, 40.0, 50.0)) -> dict:
    """Candidates per point and miss rate at smaller radii, derived from the 50 m shards (subset property)."""
    import pyarrow.parquet as pq

    rows = {r: 0 for r in radii}
    n_pts = 0
    n_pts_in_span = 0
    miss = {r: 0 for r in radii}
    miss_in_span = {r: 0 for r in radii}
    per_trip = {r: [] for r in radii}
    for tid in trips:
        d = pq.read_table(out_dir / partition / "candidates" / f"{tid}.parquet", columns=["dist_m"]).column(0).to_numpy()
        p = pq.read_table(out_dir / partition / "points" / f"{tid}.parquet", columns=["align_dist_m", "route_pos", "label_status"]).to_pandas()
        k_last = int(p.route_pos.max())
        end_off = (p.label_status.astype(str) == "off_route") & ((p.route_pos == 0) | (p.route_pos == k_last))
        n_pts += len(p)
        n_pts_in_span += int((~end_off).sum())
        for r in radii:
            rows[r] += int((d <= r).sum())
            miss[r] += int((p.align_dist_m > r).sum())
            miss_in_span[r] += int(((p.align_dist_m > r) & ~end_off).sum())
            per_trip[r].append((tid, float(((p.align_dist_m > r) & ~end_off).sum() / max(int((~end_off).sum()), 1))))
    return {
        f"r{int(r)}": {
            "candidate_rows": rows[r],
            "candidates_per_point": rows[r] / max(n_pts, 1),
            "candidate_miss_rate_all_points": miss[r] / max(n_pts, 1),
            "candidate_miss_rate_excl_route_ends": miss_in_span[r] / max(n_pts_in_span, 1),
            "per_trip_miss_rate_excl_route_ends_median": float(np.median([v for _, v in per_trip[r]])),
            "per_trip_miss_rate_excl_route_ends_max": max(per_trip[r], key=lambda kv: kv[1]),
            "trips_with_zero_miss_excl_route_ends": int(sum(v == 0 for _, v in per_trip[r])),
        }
        for r in radii
    }


def aggregate(stats: list) -> dict:
    lq = [s["label_quality"] for s in stats if "label_quality" in s]
    n_pts = sum(s["n_points"] for s in stats)

    def wavg(key, weight="n_points"):
        num = sum(q[key] * q[weight] for q in lq if key in q and not (isinstance(q[key], float) and np.isnan(q[key])))
        den = sum(q[weight] for q in lq if key in q and not (isinstance(q[key], float) and np.isnan(q[key])))
        return num / den if den else None

    agg = {
        "n_trips": len(stats),
        "n_points": n_pts,
        "n_candidate_rows": sum(s["n_candidate_rows"] for s in stats),
        "points_without_candidates": sum(s["candidate_stats"].get("points_without_candidates", 0) for s in stats),
        "n_segments_total": sum(s["n_segments"] for s in stats),
        "time_total_s": sum(s["timings"]["total_s"] + s["timings"].get("write_s", 0) for s in stats),
        "peak_mem_mb_max": max(s["peak_mem_mb"] for s in stats),
        "slowest_trip": max(stats, key=lambda s: s["timings"]["total_s"])["trip_id"],
        "slowest_trip_s": max(s["timings"]["total_s"] for s in stats),
    }
    if lq:
        tot = {k: sum(q[k] for q in lq) for k in ("n_ok", "n_boundary", "n_off_route", "off_route_at_route_ends", "off_route_mid_route",
                                                  "positives", "label_equiv_rows", "points_with_equiv_twin", "near_tie_negative_points")}
        labelled = tot["n_ok"] + tot["n_boundary"]
        agg["labels"] = {
            **tot,
            "points_labelled": labelled,
            "labelled_frac": labelled / n_pts,
            "boundary_frac_of_labelled": tot["n_boundary"] / max(labelled, 1),
            "near_tie_frac_of_labelled": tot["near_tie_negative_points"] / max(labelled, 1),
            "positive_rate_rows": tot["positives"] / max(agg["n_candidate_rows"], 1),
            "max_positives_per_point": max(q["max_positives_per_point"] for q in lq),
            "align_dist_median_m_trip_median": float(np.median([q["align_dist_median_m"] for q in lq if "align_dist_median_m" in q])),
            "align_dist_p95_m_trip_median": float(np.median([q["align_dist_p95_m"] for q in lq if "align_dist_p95_m" in q])),
            "route_coverage_in_span_mean": float(np.mean([q["route_coverage_in_span"] for q in lq])),
            "progress_abs_err_median_m_trip_median": float(np.median([q["progress_abs_err_median_m"] for q in lq if "progress_abs_err_median_m" in q])),
            "progress_err_gt25m_frac_wavg": wavg("progress_err_gt25m_frac"),
            "label_agreement_w0_wavg": wavg("label_agreement_w0"),
            "label_agreement_w30_wavg": wavg("label_agreement_w30"),
            "baseline_nearest_top1_wavg": wavg("nearest_candidate_top1_on_labelled"),
            "baseline_anchor_heuristic_top1_wavg": wavg("anchor_heuristic_top1_on_labelled"),
            "trips_route_connectivity_lt_1": sorted(s["trip_id"] for s in stats if s.get("label_quality", {}).get("route_connectivity", 1.0) < 1.0),
            "trips_offroute_gt10pct": sorted(
                s["trip_id"] for s in stats if s.get("label_quality") and s["label_quality"]["n_off_route"] > 0.1 * s["n_points"]
            ),
        }
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partitions", nargs="+", default=["train", "student_test"])
    ap.add_argument("--trips", nargs="*", default=None, help="subset of trip ids (default: all in partition)")
    ap.add_argument("--radius", type=float, default=50.0)
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--report", type=Path, default=REPORT)
    ap.add_argument("--no-consolidate", action="store_true")
    ap.add_argument("--sensitivity", nargs="*", type=float, default=[0.0, 30.0], help="alt heading weights for label-stability check")
    args = ap.parse_args()

    from src.phase3_pipeline import Phase3Config
    from src.candidate_generator import CandidateConfig

    args.out.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": Phase3Config(candidate=CandidateConfig(radius_m=args.radius)).to_dict(),
        "workers": args.workers,
        "partitions": {},
    }
    wall0 = time.perf_counter()
    for part in args.partitions:
        all_trips = sorted(p.name for p in (RAW / part).iterdir() if p.is_dir())
        trips = [t for t in all_trips if args.trips is None or t in args.trips]
        if not trips:
            continue
        # biggest first for load balancing
        trips_by_size = sorted(trips, key=lambda t: -sum(f.stat().st_size for f in (RAW / part / t).iterdir()))
        print(f"[{part}] {len(trips)} trips, {args.workers} workers, radius {args.radius} m", flush=True)
        t_part = time.perf_counter()
        stats = []
        with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as ex:
            futs = {ex.submit(_worker, t, part, args.radius, str(args.out), tuple(args.sensitivity)): t for t in trips_by_size}
            for f in as_completed(futs):
                s = f.result()
                stats.append(s)
                q = s.get("label_quality", {})
                print(
                    f"  {s['trip_id']}  pts={s['n_points']:>6}  arcs={s['n_arcs']:>6}  rows={s['n_candidate_rows']:>7}  "
                    f"t={s['timings']['total_s']:.1f}s  peak={s['peak_mem_mb']:.0f}MB  "
                    f"off_route={q.get('n_off_route', '-')}  boundary={q.get('n_boundary', '-')}  ({len(stats)}/{len(trips)})",
                    flush=True,
                )
        stats.sort(key=lambda s: s["trip_id"])
        part_rep = {"wall_s": time.perf_counter() - t_part, "aggregate": aggregate(stats), "trips": stats}
        if not args.no_consolidate:
            t = time.perf_counter()
            part_rep["files"] = consolidate(part, trips, args.out)
            part_rep["consolidate_s"] = time.perf_counter() - t
            if any("label_quality" in s for s in stats):
                part_rep["radius_sweep"] = radius_sweep(part, trips, args.out)
        report["partitions"][part] = part_rep
        a = part_rep["aggregate"]
        print(f"[{part}] done in {part_rep['wall_s']:.1f}s wall | points={a['n_points']} rows={a['n_candidate_rows']}", flush=True)

    report["wall_total_s"] = time.perf_counter() - wall0
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
