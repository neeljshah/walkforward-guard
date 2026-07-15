"""
walkforward_guard.core

Two small tools for time-series validation that actually catch leaks:

1. WalkForwardSplitter -- an expanding-window, purge-gapped CV splitter that
   ASSERTS (not just "tries") that no test-fold row precedes the end of its
   train fold, and raises LeakError immediately if it does.

2. assert_truncation_invariant -- a property check for a feature function:
   a feature computed "as of" time t, using only events up to t, must be
   IDENTICAL whether you hand it the full future-including dataset (pre-
   filtered by the caller) or a dataset that was truncated at t before the
   caller ever saw it. If those two disagree, the feature is peeking.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator

import numpy as np
import pandas as pd


class LeakError(Exception):
    """Raised the moment a fold or a feature is caught looking into the future."""


@dataclass
class Fold:
    train_idx: np.ndarray
    test_idx: np.ndarray
    train_end: pd.Timestamp
    test_start: pd.Timestamp


class WalkForwardSplitter:
    """Expanding-window walk-forward CV with a mandatory purge gap.

    Unlike sklearn's ``TimeSeriesSplit``, every fold this yields has already
    been asserted leak-free before it reaches your training loop -- there is
    no way to silently consume a bad fold.

    Parameters
    ----------
    n_splits:
        Number of forward folds to produce.
    purge_gap:
        A ``pd.Timedelta`` (or anything ``pd.Timedelta`` accepts) removed
        from the end of train, immediately before the test fold starts.
        Use this to model label latency / settlement delay -- the gap a
        real signal would actually have to survive in production.
    min_train_fraction:
        Minimum fraction of the dataset reserved for the first train fold,
        so early folds are not trained on a handful of rows.
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: str | pd.Timedelta = "0D",
        min_train_fraction: float = 0.3,
    ) -> None:
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if not 0.0 < min_train_fraction < 1.0:
            raise ValueError("min_train_fraction must be in (0, 1)")
        self.n_splits = n_splits
        self.purge_gap = pd.Timedelta(purge_gap)
        self.min_train_fraction = min_train_fraction

    def split(self, df: pd.DataFrame, time_col: str) -> Iterator[Fold]:
        """Yield leak-checked ``Fold`` objects in chronological order.

        Raises
        ------
        LeakError
            If any fold's max train timestamp is not strictly before its
            min test timestamp once the purge gap is applied -- e.g. the
            input was shuffled, or duplicate/out-of-order timestamps let a
            test row slip into train.
        """
        times = df[time_col]
        if not times.is_monotonic_increasing:
            raise LeakError(
                f"{time_col!r} is not sorted ascending -- sort the frame "
                "before splitting, a shuffled index is the #1 way folds leak."
            )

        n = len(df)
        first_train_end = int(n * self.min_train_fraction)
        remaining = n - first_train_end
        if remaining < self.n_splits:
            raise ValueError("not enough rows after min_train_fraction for n_splits")
        test_size = remaining // self.n_splits

        order = np.arange(n)
        train_end = first_train_end
        for k in range(self.n_splits):
            test_start = train_end
            test_end = n if k == self.n_splits - 1 else test_start + test_size
            if test_start >= test_end:
                continue

            train_idx = order[:train_end]
            test_idx = order[test_start:test_end]

            # Apply the purge gap: drop train rows too close to the test window.
            cutoff = times.iloc[test_start] - self.purge_gap
            keep = times.iloc[train_idx].values < cutoff
            train_idx = train_idx[keep]

            if len(train_idx) == 0:
                train_end = test_end
                continue

            fold = Fold(
                train_idx=train_idx,
                test_idx=test_idx,
                train_end=times.iloc[train_idx].max(),
                test_start=times.iloc[test_idx].min(),
            )
            self._assert_no_leak(fold)
            yield fold
            train_end = test_end

    @staticmethod
    def _assert_no_leak(fold: Fold) -> None:
        """The single assertion this whole package exists to run."""
        if not fold.train_end < fold.test_start:
            raise LeakError(
                f"fold leak: max(train time)={fold.train_end} is not strictly "
                f"before min(test time)={fold.test_start}. A train row sits "
                "at or after the test window -- do not train on this fold."
            )


def assert_truncation_invariant(
    featurize_fn: Callable[[pd.DataFrame, pd.Timestamp], object],
    events: pd.DataFrame,
    t: pd.Timestamp,
    time_col: str = "ts",
) -> None:
    """Assert a feature function is truncation-invariant at time ``t``.

    ``featurize_fn(events_df, t)`` is expected to compute a feature "as of"
    ``t`` using only rows with ``time_col <= t``. This check calls it twice:

    1. with the full ``events`` frame (which may contain rows after ``t``)
    2. with ``events`` pre-truncated to rows with ``time_col <= t``

    A correct as-of feature ignores future rows entirely, so both calls must
    return the identical value. If they differ, ``featurize_fn`` is reading
    something from after ``t`` -- classic lookahead leakage that a
    single-split backtest will never surface, because a single split never
    tests the same cutoff twice with different amounts of "future" attached.

    Raises
    ------
    LeakError
        If the two results differ.
    """
    truncated = events[events[time_col] <= t]

    full_result = featurize_fn(events, t)
    truncated_result = featurize_fn(truncated, t)

    if not _values_equal(full_result, truncated_result):
        raise LeakError(
            f"truncation invariance violated at t={t}: featurize_fn(full)="
            f"{full_result!r} != featurize_fn(truncated)={truncated_result!r}. "
            "The function is using rows after t -- it is leaking the future."
        )


def _values_equal(a: object, b: object) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        if np.isnan(a) and np.isnan(b):
            return True
        return np.isclose(a, b, equal_nan=True)
    try:
        return bool(a == b)
    except ValueError:
        return np.array_equal(np.asarray(a), np.asarray(b))
