"""
Evaluation of ML-only vs HMM/Viterbi predictions (ground truth used ONLY here, after inference).

Metrics
* top1_accuracy          share of labelled GPS points whose predicted arc is the reference arc
                         (same definition and denominator as Phase 4: label_status ok/boundary)
* top1_accuracy_equiv    same, also accepting a parallel duplicate (twin) of the reference arc
* continuity_direct      consecutive points (same time segment) on the same or directly connected arcs
* continuity_reach3      next arc reachable within <= 3 arcs (as in Phase 4)
* route_precision/recall/f1 (+ length-weighted) of the decoded arc set vs the reference route

CLI (all student-test trips, writes results/metrics/hmm_evaluation.json):
    python -m src.phase5_hmm.evaluate_hmm
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def point_accuracy(pred_arc: np.ndarray, cand_labels: pd.DataFrame, labelled: np.ndarray) -> Dict:
    pts = np.flatnonzero(labelled)
    n = len(pts)
    if n == 0:
        return {"points": 0, "correct": 0, "top1_accuracy": None, "top1_accuracy_equiv": None}
    pred = pd.DataFrame({"point_idx": pts.astype(np.int64), "arc_id": pred_arc[pts].astype(np.int64)})
    lab = cand_labels[["point_idx", "arc_id", "label", "label_equiv"]].astype(
        {"point_idx": np.int64, "arc_id": np.int64}
    ).drop_duplicates(["point_idx", "arc_id"])
    m = pred.merge(lab, on=["point_idx", "arc_id"], how="left")
    hit = m["label"].fillna(0).to_numpy() == 1
    eq = hit | (m["label_equiv"].fillna(0).to_numpy() == 1)
    return {
        "points": int(n),
        "correct": int(hit.sum()),
        "top1_accuracy": float(hit.mean()),
        "top1_accuracy_equiv": float(eq.mean()),
    }


def _hops_le(fr: np.ndarray, out_ptr: np.ndarray, out_arcs: np.ndarray, to: np.ndarray, a: int, b: int, max_hops: int) -> bool:
    frontier, seen = {a}, {a}
    for _ in range(max_hops):
        nxt = set()
        for x in frontier:
            n = to[x]
            for y in out_arcs[out_ptr[n]:out_ptr[n + 1]].tolist():
                if y == b:
                    return True
                if y not in seen:
                    seen.add(y)
                    nxt.add(y)
        frontier = nxt
        if not frontier:
            break
    return False


def continuity(pred_arc: np.ndarray, segment_id: np.ndarray, fr: np.ndarray, to: np.ndarray, max_hops: int = 3) -> Dict:
    p = np.flatnonzero(pred_arc >= 0)
    if len(p) < 2:
        return {"pairs": 0, "continuity_direct": None, "continuity_reach3": None, "arc_switches": 0}
    a = pred_arc[p]
    ok = (np.diff(p) == 1) & (segment_id[p][1:] == segment_id[p][:-1])
    x, y = a[:-1][ok], a[1:][ok]
    direct = (x == y) | (to[x] == fr[y])
    n_nodes = int(max(fr.max(), to.max()) + 1)
    out_arcs = np.argsort(fr, kind="stable")
    out_ptr = np.r_[0, np.cumsum(np.bincount(fr, minlength=n_nodes))]
    reach = direct.copy()
    cache: Dict = {}
    for i in np.flatnonzero(~direct):
        k = (int(x[i]), int(y[i]))
        if k not in cache:
            cache[k] = _hops_le(fr, out_ptr, out_arcs, to, k[0], k[1], max_hops)
        reach[i] = cache[k]
    n = int(ok.sum())
    return {
        "pairs": n,
        "continuity_direct": float(direct.mean()) if n else None,
        "continuity_reach3": float(reach.mean()) if n else None,
        "arc_switches": int((x != y).sum()),
    }


def compress(seq: Sequence[int]) -> list:
    out = []
    for a in seq:
        a = int(a)
        if a < 0:
            continue
        if not out or out[-1] != a:
            out.append(a)
    return out


def route_overlap(pred_arcs: Sequence[int], ref_route: Sequence[int], length_m: np.ndarray) -> Dict:
    p, r = set(int(a) for a in pred_arcs if a >= 0), set(int(a) for a in ref_route)
    if not p or not r:
        return {"route_precision": None, "route_recall": None, "route_f1": None}
    inter = p & r
    prec, rec = len(inter) / len(p), len(inter) / len(r)
    lp = float(length_m[list(p)].sum())
    lr = float(length_m[list(r)].sum())
    li = float(length_m[list(inter)].sum()) if inter else 0.0
    lprec, lrec = (li / lp if lp else 0.0), (li / lr if lr else 0.0)
    f1 = lambda a, b: (2 * a * b / (a + b)) if (a + b) else 0.0  # noqa: E731
    return {
        "route_precision": prec,
        "route_recall": rec,
        "route_f1": f1(prec, rec),
        "route_length_precision": lprec,
        "route_length_recall": lrec,
        "route_length_f1": f1(lprec, lrec),
    }


def aggregate(per_trip: Sequence[Dict]) -> Dict:
    """Pool point-level counts over trips; average route-level metrics."""
    out: Dict = {}
    for method in ("ml_only", "hmm"):
        pts = sum(t["metrics"][method]["points"] for t in per_trip)
        cor = sum(t["metrics"][method]["correct"] for t in per_trip)
        eqv = sum(t["metrics"][method]["top1_accuracy_equiv"] * t["metrics"][method]["points"] for t in per_trip)
        pairs = sum(t["metrics"][method]["pairs"] for t in per_trip)
        d = sum((t["metrics"][method]["continuity_direct"] or 0) * t["metrics"][method]["pairs"] for t in per_trip)
        r3 = sum((t["metrics"][method]["continuity_reach3"] or 0) * t["metrics"][method]["pairs"] for t in per_trip)
        sw = sum(t["metrics"][method]["arc_switches"] for t in per_trip)
        route = {k: float(np.mean([t["metrics"][method][k] for t in per_trip if t["metrics"][method][k] is not None]))
                 for k in ("route_precision", "route_recall", "route_f1", "route_length_f1")}
        out[method] = {
            "points": pts,
            "top1_accuracy": cor / pts if pts else None,
            "top1_accuracy_equiv": eqv / pts if pts else None,
            "continuity_direct": d / pairs if pairs else None,
            "continuity_reach3": r3 / pairs if pairs else None,
            "arc_switches": sw,
            **{f"mean_{k}": v for k, v in route.items()},
        }
    return out


def main(argv: Optional[Sequence[str]] = None) -> Dict:
    from src.phase6_pipeline.end_to_end_matcher import PathSnapPipeline, list_trips

    ap = argparse.ArgumentParser(description="Evaluate ML-only vs HMM on student-test trips")
    ap.add_argument("--trips", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "metrics" / "hmm_evaluation.json")
    args = ap.parse_args(argv)

    pipe = PathSnapPipeline()
    trips = args.trips or list_trips("student_test")
    per_trip = []
    t0 = time.perf_counter()
    for tid in trips:
        r = pipe.run(tid, partition="student_test", evaluate=True, include_geometry=False)
        per_trip.append({k: r[k] for k in ("trip_id", "n_points", "n_candidate_rows", "metrics", "timing_ms", "hmm_diagnostics")})
        m = r["metrics"]
        print(f"{tid}: ML-only {m['ml_only']['top1_accuracy']:.4f}  HMM {m['hmm']['top1_accuracy']:.4f}  "
              f"reach3 {m['ml_only']['continuity_reach3']:.4f}->{m['hmm']['continuity_reach3']:.4f}  "
              f"{r['timing_ms']['inference_total']:.0f} ms", flush=True)
    result = {
        "partition": "student_test",
        "model_file": pipe.model_file,
        "hmm_config": pipe.hmm_config.to_dict(),
        "trips": len(per_trip),
        "overall": aggregate(per_trip),
        "per_trip": per_trip,
        "wall_s": time.perf_counter() - t0,
        "note": "Ground truth is used only after inference, for these metrics. HMM settings are fixed a priori (not tuned on these trips).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    o = result["overall"]
    print(f"OVERALL ML-only {o['ml_only']['top1_accuracy']:.4f}  HMM {o['hmm']['top1_accuracy']:.4f}  -> {args.out}")
    return result


if __name__ == "__main__":
    main()
