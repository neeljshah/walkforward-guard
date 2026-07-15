# walkforward-guard

An assertion-level leak guard for time-series cross-validation. Not a
model, not a strategy -- a small piece of validation infrastructure that
makes it structurally hard to fool yourself about how good a time-series
model is.

Distilled out of a larger multi-sport forecasting system, where a single
undetected leak in a backtest can make a null result look like an edge. This
repo keeps only the validation piece, rebuilt against a generic synthetic
time series -- no sports data, no proprietary features.

## Why k-fold on time series lies to you

Standard k-fold cross-validation assumes rows are exchangeable. Time-series
rows are not: row 500 was measured after row 499 learned something about the
world that row 499 didn't have yet. Shuffle-then-split lets your model
"train" on data from next month and "test" on data from last month --
information that was never available at prediction time leaks backward into
training, and your validation score quietly stops meaning anything.

The usual fix -- "just use a time-ordered split" -- is necessary but not
sufficient. Two more failure modes slip through even a chronologically
ordered split:

- **A single split window.** One train/test boundary is one sample size of
  one. A lucky (or unlucky) cutoff makes a result look robust when it's an
  artifact of that one boundary.
- **Feature leakage inside a "correct" split.** The split can be perfectly
  time-ordered and the *feature engineering* can still read the future --
  a centered rolling window, a `groupby().transform()` that sees the whole
  group, a join on a table that wasn't itself truncated. The split being
  right tells you nothing about whether the features feeding it are right.

## What this package does about it

**1. `WalkForwardSplitter`** -- expanding-window walk-forward CV with a
purge gap. Every fold it yields has already been asserted leak-free before
your training loop ever sees it:

```python
max(train_timestamps) < min(test_timestamps)
```

If that assertion fails -- a shuffled frame, duplicate timestamps, an
off-by-one in the split math -- it raises `LeakError` immediately, at the
fold, not three weeks later when someone asks why the backtest looks too
good.

**2. `assert_truncation_invariant`** -- a property test for a feature
function. A correct "as of time `t`" feature must return the identical
value whether you hand it the full future-including dataset or a dataset
truncated at `t` before it ever runs:

```python
featurize(full_data, t) == featurize(full_data[full_data.ts <= t], t)
```

If it doesn't, the feature is reading something after `t` -- and a single
train/test split will never catch this, because a single split never
re-runs the same cutoff with two different amounts of "future" attached to
see if the answer changes.

## Quickstart

```python
import pandas as pd
from walkforward_guard import WalkForwardSplitter, assert_truncation_invariant, LeakError

df = pd.DataFrame({
    "ts": pd.date_range("2026-01-01", periods=200, freq="D"),
    "value": range(200),
})

# 1. Split with a leak guard baked in.
splitter = WalkForwardSplitter(n_splits=5, purge_gap="2D")
for fold in splitter.split(df, time_col="ts"):
    train, test = df.iloc[fold.train_idx], df.iloc[fold.test_idx]
    # ... fit on train, score on test ...

# 2. Check a feature function can't see the future.
def trailing_mean(events, t):
    return events[events["ts"] <= t]["value"].tail(5).mean()

assert_truncation_invariant(trailing_mean, df, df["ts"].iloc[100])  # passes

def leaky_mean(events, t):
    return events["value"].mean()  # bug: ignores t entirely

try:
    assert_truncation_invariant(leaky_mean, df, df["ts"].iloc[100])
except LeakError as e:
    print("caught:", e)
```

## How this catches real leaks

`examples/demo_catch_a_leak.py` plants both failure modes on a synthetic
random-walk series and shows the guard catching each one:

1. **Shuffled input.** The dataset is shuffled (as if someone used a random
   `train_test_split` instead of a time split) and handed to
   `WalkForwardSplitter`. It refuses to yield a single fold and raises
   `LeakError` before any model would have trained on it.
2. **Centered rolling window.** A "trailing 5-day mean" feature is
   implemented with `rolling("5D", center=True)` instead of a trailing-only
   window -- an easy, common mistake. `assert_truncation_invariant` catches
   the mismatch between the full-data and truncated-data results and raises
   `LeakError`, while the corrected trailing-only version passes clean.

Run it: `python examples/demo_catch_a_leak.py`. It exits `1` once both
planted leaks are caught -- in this demo, a "successful" run looks like a
failing CI check, which is exactly what you want a leak guard to do in a
real pipeline.

## Install / test

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q
```

## Scope

This is validation tooling, not a forecasting model and not a claim of
predictive edge. It does one job: make it hard to accidentally lie to
yourself about whether a time-series model generalizes forward in time.

## Author

Neel Shah -- [neeljshah22@gmail.com](mailto:neeljshah22@gmail.com) --
[github.com/neeljshah](https://github.com/neeljshah)

## License

MIT, see [LICENSE](LICENSE).
