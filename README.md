# walkforward-guard

Create expanding-window time-series splits with a purge gap, and check whether a
feature changes when future rows are removed. These checks make timestamp
ordering and lookahead errors visible before model training or evaluation.

The package depends on NumPy and pandas. It supplies validation checks, not a
forecasting model or proof that a complete pipeline is free of leakage.

## Install

Requires Python 3.10 or later.

```bash
git clone https://github.com/neeljshah/walkforward-guard.git
cd walkforward-guard
python -m pip install -e ".[dev]"
```

The `dev` extra installs pytest; use `-e .` for the library alone.

## Runnable example

```bash
python examples/demo_catch_a_leak.py
```

The demo uses a seeded synthetic time series. It deliberately plants shuffled
folds and a centered rolling-window feature, and prints:

```text
--- Failure mode 1: shuffled timestamps fed to the splitter ---
CAUGHT: 'ts' is not sorted ascending -- sort the frame before splitting, a shuffled index is the #1 way folds leak.

--- Failure mode 2: centered rolling window (reads the future) ---
CAUGHT: truncation invariance violated at t=2026-04-11 00:00:00: featurize_fn(full)=9.054609396769228 != featurize_fn(truncated)=8.744403704002325. The function is using rows after t -- it is leaking the future.
Sanity check: the honest trailing-window feature passes clean...
PASSED (no leak) as expected.

=== summary ===
shuffled-folds leak caught:      True
centered-window leak caught:     True

Both planted leaks were caught by the guard. Exiting 1 (nonzero) -- that's the guard doing its job.
```

**Expected exit code: 1.** Both planted leaks were caught. In this particular
demo, exit code 0 means at least one planted leak was missed. The unit-test
command below uses the normal convention: exit code 0 means tests passed.

## Python API

```python
import pandas as pd
from walkforward_guard import WalkForwardSplitter, assert_truncation_invariant

frame = pd.DataFrame({
    "ts": pd.date_range("2026-01-01", periods=100, freq="D", tz="UTC"),
    "value": range(100),
})
folds = list(WalkForwardSplitter(n_splits=4, purge_gap="5D").split(frame, "ts"))
print(len(folds))
print(all(fold.train_end < fold.test_start for fold in folds))

def trailing_mean(events, cutoff):
    return float(events.loc[events["ts"] <= cutoff, "value"].tail(5).mean())

assert_truncation_invariant(trailing_mean, frame, frame["ts"].iloc[50])
```

```text
4
True
```

`Fold.train_idx` and `Fold.test_idx` are positional indices for `frame.iloc`.
The splitter requires ascending timestamps, removes training rows at or after
`test_start - purge_gap`, and checks `max(train_time) < min(test_time)` before
yielding each fold. Use a nonnegative purge gap that reflects your label and
feature availability delays. A fold with no training rows left after purging is
skipped, so a large gap can produce fewer than `n_splits` folds.

`assert_truncation_invariant` compares the supplied function's result on the full
frame with its result on rows whose timestamps are at or before the cutoff. A
mismatch raises `LeakError`. Scalar floats use NumPy's default `isclose`
tolerance; other results use equality or array equality.

## Limits

Use consistent datetime types and timezones, deterministic feature functions,
and timestamps that represent when information was actually available. The
splitter checks the timestamps it receives; it cannot validate external joins,
label construction or preprocessing fitted outside the training fold. A
truncation check covers the supplied function, dataset and cutoff, so run it at
multiple representative cutoffs. A passing check is not a complete leakage
audit.

## Tests

```bash
python -m pytest tests/test_core.py -q
```

The seven tests cover forward ordering, purge gaps, UTC-aware timestamps,
shuffled input, invalid configuration and honest versus leaky feature functions.
GitHub Actions runs this file with a two-minute test timeout.

## Context and license

[The CourtVision evidence site](https://neeljshah.github.io/court-vision/)
contains the forecasting results and public retraction record. Its moneyline
models land near the devigged close without beating it; totals and tennis trail.

[MIT](LICENSE), copyright Neel Shah 2026.
