"""
demo_catch_a_leak.py

Builds a synthetic time series, plants two classic leaks, and shows
walkforward_guard catching both. This is a demo of the guard firing, so a
"successful" run looks like a CI failure: it exits 1 (nonzero) once both
leaks are caught, exactly as it would in a real pipeline where the guard
just stopped you from training on a leaky fold. It only exits 0 -- meaning
"nothing was caught" -- if the guard failed to detect a planted leak, which
would itself be a bug worth investigating.

Run: python examples/demo_catch_a_leak.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from walkforward_guard import LeakError, WalkForwardSplitter, assert_truncation_invariant


def make_events(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n, freq="D")
    value = rng.normal(size=n).cumsum()
    return pd.DataFrame({"ts": ts, "value": value})


def failure_mode_1_shuffled_folds() -> bool:
    """Failure mode 1: someone shuffles the frame before CV (e.g. a random
    train_test_split instead of a time split). The splitter must refuse."""
    print("\n--- Failure mode 1: shuffled timestamps fed to the splitter ---")
    events = make_events().sample(frac=1.0, random_state=1).reset_index(drop=True)
    splitter = WalkForwardSplitter(n_splits=4, purge_gap="1D")
    try:
        list(splitter.split(events, time_col="ts"))
    except LeakError as e:
        print(f"CAUGHT: {e}")
        return True
    print("NOT CAUGHT -- this is a bug in the guard.")
    return False


def featurize_leaky(events: pd.DataFrame, t: pd.Timestamp) -> float:
    """Plants the leak: a 'trailing 5-day mean as of t' that was actually
    implemented with a CENTERED rolling window, so it silently pulls in
    values from after t. This is the single most common way a feature
    pipeline leaks -- a rolling/window op with the wrong alignment."""
    df = events.sort_values("ts")
    df = df.set_index("ts")
    centered = df["value"].rolling("5D", center=True).mean()
    idx = df.index.get_indexer([t], method="nearest")[0]
    return float(centered.iloc[idx])


def featurize_honest(events: pd.DataFrame, t: pd.Timestamp) -> float:
    """The fixed version: trailing-only window, as-of t."""
    df = events[events["ts"] <= t].sort_values("ts")
    return float(df["value"].tail(5).mean())


def failure_mode_2_centered_window() -> bool:
    """Failure mode 2: a feature function uses a centered rolling window
    instead of a trailing one, so it reads rows after the as-of cutoff."""
    print("\n--- Failure mode 2: centered rolling window (reads the future) ---")
    events = make_events()
    t = events["ts"].iloc[100]

    caught = False
    try:
        assert_truncation_invariant(featurize_leaky, events, t)
        print("NOT CAUGHT -- this is a bug in the guard.")
    except LeakError as e:
        print(f"CAUGHT: {e}")
        caught = True

    print("Sanity check: the honest trailing-window feature passes clean...")
    assert_truncation_invariant(featurize_honest, events, t)  # must not raise
    print("PASSED (no leak) as expected.")
    return caught


def main() -> int:
    ok1 = failure_mode_1_shuffled_folds()
    ok2 = failure_mode_2_centered_window()

    print("\n=== summary ===")
    print(f"shuffled-folds leak caught:      {ok1}")
    print(f"centered-window leak caught:     {ok2}")

    if ok1 and ok2:
        print("\nBoth planted leaks were caught by the guard. Exiting 1 "
              "(nonzero) -- that's the guard doing its job.")
        return 1
    print("\nAt least one planted leak was NOT caught -- guard regression. Exiting 0.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
