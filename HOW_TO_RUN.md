# How to train and test this model

A focused walkthrough of the 3 commands that train and test the strategy.
For the full picture (artifacts, assumptions, tensor shapes, limitations),
see `README.md` — this file is just the "do this" version.

## 0. Install once

```bash
cd Practice
pip install torch pandas numpy scikit-learn hmmlearn pyyaml pyarrow matplotlib pytest
```

## 1. Train

```bash
python cli.py train --config config_demo.yaml --run-id mytest1
```

What this does:
- Generates synthetic OHLC data for tickers AAA/BBB/CCC (`synthetic_data.py`
  — **not real market data**, see "Which config do I use" below).
- Splits the date range into one expanding-window fold: 60% train / 20% dev
  / 20% test.
- Builds technical + regime (HMM) + seasonal features for each ticker.
- Trains the GRU take/skip classifier on the train window.
- Picks a probability threshold on the dev window (never on test — that
  would be leakage) that maximizes dev-set expectancy.
- Saves every artifact to `runs/mytest1/`: `model.pt`, `regime_model.json`,
  `feature_schema.json`, `scaler.json`, `run_metadata.json`,
  `experiment_log.jsonl`.

You'll see lines like `Model is not converging. Current: ... Delta is ...`
from `hmmlearn` — that's the regime model's Baum-Welch fit hitting its
iteration cap. Informational, not fatal; the fitted parameters are still
saved and used.

Expect output like:
```
train: wrote artifacts to runs\mytest1 (threshold=0.5, dev_expectancy=0.0298...)
```

## 2. Test (evaluate)

```bash
python cli.py evaluate --run-id mytest1
```

What this does:
- Reloads the model, regime model, and threshold saved in step 1.
- Scores every candidate trade in the **held-out test window** (never seen
  during training or threshold selection).
- Runs the Step 9 portfolio backtest on the accepted trades: position
  sizing by fixed risk budget, admission control (no leverage, position
  cap, concurrent-trade cap, aggregate-risk cap, no duplicate-ticker
  overlap), a real equity curve built from cash flows.
- Writes `runs/mytest1/reports/fold_metrics.csv` and three plots:
  `equity_curve.png`, `drawdown.png`, `calibration.png`.

`fold_metrics.csv` columns, and what they mean:

| Column | Meaning |
|---|---|
| `n_test_trades` | candidates that fired in the test window |
| `n_admitted_portfolio` | how many actually got capital allocated (after risk/position caps) |
| `selected_threshold` | the dev-selected probability cutoff, reused here unchanged |
| `test_accuracy` | classification accuracy of "would take" vs the trade's actual label |
| `test_expectancy` | mean `net_return` across accepted trades |
| `label_precision` | fraction of accepted trades where `label == 1` (hit 2R target) |
| `profit_rate` | fraction of accepted trades where `net_return > 0` — **different from `label_precision` on purpose**: a trade can be profitable without hitting the full target, or vice versa |
| `final_equity` | ending portfolio value, starting from `starting_cash` |
| `max_drawdown` | worst peak-to-trough drop in the equity curve |
| `benchmark_return` | buy-and-hold return on the benchmark over the identical test window — the thing the strategy has to beat |

## 3. Check live-ish signals

```bash
python cli.py signals --run-id mytest1 --as-of 2019-12-20
```

Reloads the trained model and reports any candidate signal firing in the
10 sessions before `--as-of`, with its `take_prob` and whether it clears
the saved threshold:

```
CCC    2019-12-13    take_prob=0.5100    threshold=0.5000    would_take=True
```

If nothing fired in that window, it says so instead of printing nothing
silently.

## Which config do I use?

- **`config_demo.yaml`** (used above) — 3 synthetic tickers, a longer
  history, a lower `min_trades_for_threshold`. Reliably produces a trained
  model. Use this to learn the commands.
- **`config.yaml`** — mirrors the project's reference spec exactly (2
  tickers, `min_trades_for_threshold: 20`). On the synthetic data this
  project currently ships with, this can legitimately print
  `insufficient data` instead of training — that's the safety guardrail
  working as designed (too few dev-window trades to pick a threshold
  responsibly), not a bug.

Neither config points at real market data yet — see `README.md`'s
Limitations section for what that would take.

## Re-running from scratch

Delete `runs/<run-id>/` and repeat steps 1-3 with the same `--run-id` to
retrain from a clean slate. Nothing under `runs/` is read by `train` —
each training run regenerates its data and model independently.

## One command that answers "did I break anything"

```bash
pytest -q
```

21 tests, including `test_end_to_end_smoke.py`, which runs all three
commands above itself, end to end, in about 5 seconds. Run this after any
change to any module.
