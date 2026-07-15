import numpy as np
import pandas as pd
import pytest

from walkforward_guard import LeakError, WalkForwardSplitter, assert_truncation_invariant


def make_df(n=100, seed=0):
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="D")
    return pd.DataFrame({"ts": ts, "value": rng.normal(size=n)})


def test_splitter_ordering_is_strictly_forward():
    df = make_df()
    splitter = WalkForwardSplitter(n_splits=4, purge_gap="0D")
    folds = list(splitter.split(df, time_col="ts"))
    assert len(folds) == 4
    for fold in folds:
        assert fold.train_end < fold.test_start
        assert max(fold.train_idx) < min(fold.test_idx)


def test_splitter_purge_gap_removes_train_rows_near_boundary():
    df = make_df()
    no_gap = list(WalkForwardSplitter(n_splits=4, purge_gap="0D").split(df, "ts"))
    with_gap = list(WalkForwardSplitter(n_splits=4, purge_gap="5D").split(df, "ts"))
    for a, b in zip(no_gap, with_gap):
        assert len(b.train_idx) <= len(a.train_idx)
        assert b.test_start - b.train_end >= pd.Timedelta("5D")


def test_leak_error_on_shuffled_dates():
    df = make_df().sample(frac=1.0, random_state=1).reset_index(drop=True)
    splitter = WalkForwardSplitter(n_splits=4)
    with pytest.raises(LeakError):
        list(splitter.split(df, time_col="ts"))


def test_splitter_rejects_bad_config():
    with pytest.raises(ValueError):
        WalkForwardSplitter(n_splits=1)
    with pytest.raises(ValueError):
        WalkForwardSplitter(min_train_fraction=1.5)


def _featurize_honest(events: pd.DataFrame, t: pd.Timestamp) -> float:
    df = events[events["ts"] <= t]
    return float(df["value"].tail(3).mean())


def _featurize_leaky(events: pd.DataFrame, t: pd.Timestamp) -> float:
    # bug: ignores t and just uses the whole (possibly future-containing) frame
    return float(events["value"].mean())


def test_truncation_invariant_passes_for_honest_feature():
    df = make_df()
    t = df["ts"].iloc[50]
    assert_truncation_invariant(_featurize_honest, df, t)  # must not raise


def test_truncation_invariant_raises_for_leaky_feature():
    df = make_df()
    t = df["ts"].iloc[50]
    with pytest.raises(LeakError):
        assert_truncation_invariant(_featurize_leaky, df, t)
