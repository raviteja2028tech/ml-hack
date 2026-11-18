"""
Loader for the Phase 3 datasets (read-only helper for later phases).

    from src.phase3_dataset import load_candidates
    train = load_candidates("train")                      # labelled points only (default)
    test  = load_candidates("student_test", columns=[...])

``labelled_only=True`` (default) keeps only GPS points whose reference arc is in
their candidate set (label_status in {ok, boundary}). Off-route points are
dropped. In trips 00000002 / 00000033 they are real driving on mapped roads that
the reference .route omits, so the true road is probably among their candidates
but carries label 0. Training on them would inject false negatives.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import pandas as pd
import pyarrow.dataset as ds

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed" / "phase3"
PARTITIONS = ("train", "student_test")


def load_candidates(
    partition: str,
    columns: Optional[List[str]] = None,
    labelled_only: bool = True,
    trips: Optional[Sequence[str]] = None,
    drop_boundary: bool = False,
) -> pd.DataFrame:
    if partition not in PARTITIONS:
        raise ValueError(f"partition must be one of {PARTITIONS}")
    path = PROCESSED / f"{partition}_candidates.parquet"
    dset = ds.dataset(path, format="parquet")
    filt = None
    if labelled_only:
        allowed = ["ok"] if drop_boundary else ["ok", "boundary"]
        filt = ds.field("label_status").isin(allowed)
    if trips is not None:
        f2 = ds.field("trip_id").isin(list(trips))
        filt = f2 if filt is None else filt & f2
    cols = None
    if columns is not None:
        cols = list(dict.fromkeys(columns))
    return dset.to_table(columns=cols, filter=filt).to_pandas()


def load_points(partition: str, trips: Optional[Sequence[str]] = None) -> pd.DataFrame:
    path = PROCESSED / f"{partition}_points.parquet"
    dset = ds.dataset(path, format="parquet")
    filt = ds.field("trip_id").isin(list(trips)) if trips is not None else None
    return dset.to_table(filter=filt).to_pandas()
