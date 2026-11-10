# PytorchStrategy — Practice/

A regime-aware, GRU-filtered pullback strategy, built and tested in ten steps
(`STEP_1_*.md` through `STEP_10_PACKAGING.md`). This README is the thing meant
to outlive those step docs — read this if you just want to run the pipeline.

## What this project actually tests

The research question is: **does regime/seasonal information help a simple
pullback entry filter, compared to not using it at all?** Building the
pipeline correctly (this repo) is necessary to answer that question but is
**not** the answer itself — see "What this has NOT proven" below.

## Install

```bash
pip install torch pandas numpy scikit-learn hmmlearn pyyaml pyarrow matplotlib pytest
```

## Run the test suite

```bash
cd Practice
pytest -q
```

21 tests across all ten steps, plus the end-to-end smoke test, should pass.
`test_end_to_end_smoke.py` is the single regression gate — run it after any
change to any module.

## Run the CLI

Four subcommands, in order:

```bash
python cli.py preprocess --config config.yaml --out runs/demo/preprocess/
python cli.py train       --config config.yaml --run-id demo
python cli.py evaluate    --run-id demo
python cli.py signals     --run-id demo --as-of 2024-06-01
```

- **`preprocess`**: builds every ticker's candidate signals + simulated trade
  outcomes (Steps 4-5), writes them to `<out>/candidates.parquet`.
- **`train`**: fits one expanding-window fold (60% train / 20% dev / 20% test
  split of the configured date range), trains the GRU take/skip classifier,
  picks a dev-set threshold, and writes every artifact under `runs/<run-id>/`.
- **`evaluate`**: reloads a trained run, scores the test window, runs the
  Step 9 portfolio backtest on accepted trades, and writes
  `runs/<run-id>/reports/` (`fold_metrics.csv`, `equity_curve.png`,
  `drawdown.png`, `calibration.png`).
- **`signals`**: reloads a trained run and reports, for any candidate signal
  firing on or just before `--as-of`, the model's `take_prob` and whether
  it clears the saved threshold.

### Try it yourself

`config.yaml` mirrors `STEP_10_PACKAGING.md`'s reference config exactly
(2 tickers, `min_trades_for_threshold: 20`). On the synthetic data this
project currently runs on (see Limitations), that config can legitimately
report **"insufficient data"** at train time — that is the guardrail
working correctly, not a bug: it means fewer than 20 dev-window trades
fired, so no threshold could be selected responsibly.

For a demo that reliably produces a trained model and a full
`evaluate`/`signals` walkthrough, use `config_demo.yaml` instead (3 tickers,
a longer synthetic history, a lower `min_trades_for_threshold`):

```bash
python cli.py train    --config config_demo.yaml --run-id demo
python cli.py evaluate --run-id demo
python cli.py signals  --run-id demo --as-of 2019-12-20
```

This should finish in a few seconds on CPU and populate every file listed
under "Artifacts written per run" below.

## Artifacts written per run (`runs/<run-id>/`)

| File | Contents |
|---|---|
| `config_used.yaml` | exact config for this run, for reproducibility |
| `run_metadata.json` | seed, package versions, UTC timestamp, fold boundaries, selected threshold, dev expectancy |
| `model.pt` | `TakeSkipGRU.state_dict()` only (not the whole module) |
| `scaler.json` | mean/std per feature column, fit on train only — **informational only**, see Limitations |
| `regime_model.json` | each ticker's frozen `CausalHMM` params (`pi`, `A`, `means`, `vars`) as plain JSON lists |
| `feature_schema.json` | the exact ordered feature-column list used to build training tensors, plus `lookback` |
| `experiment_log.jsonl` | Step 8's append-only JSON-lines log, one row per CLI `train` run |
| `reports/fold_metrics.csv` | one row: trade counts, accuracy, expectancy, label precision, profit rate, final equity, max drawdown, benchmark return |
| `reports/equity_curve.png` | the Step 9 portfolio equity curve |
| `reports/drawdown.png` | drawdown from running peak equity |
| `reports/calibration.png` | realized win-rate by `take_prob` bucket on the test window |

## Non-negotiable assumptions (pulled from Steps 1-9)

- Long-only, no leverage (`notional <= cash_available`, always).
- Entry at next session's open after a signal fires (t+1-open), never same-bar.
- Fixed max holding period (`max_holding_sessions`, default 7 sessions).
- Daily bars only — no intraday data anywhere in this pipeline.
- Commission + slippage applied in both directions (`commission_bps`, `slippage_bps`).
- A fresh `CausalHMM` is refit per fold, on train-window data only, then frozen —
  it is never refit using dev or test observations.
- Position sizing is a fixed fraction of **current** equity (`risk_pct_per_trade`),
  recomputed at each admission, sized from the initial stop distance.
- Deterministic same-day candidate ranking: `take_prob` descending, ticker
  name ascending as the tie-break.
- No two open positions in the same ticker at once — a repeat signal in an
  already-open ticker is skipped outright, not queued.

## Tensor shapes at every boundary

- Feature window into the GRU: `[lookback, F]` per sample (`F` = number of
  switched-on feature-block columns — see `feature_schema.json`).
- Batched input to `TakeSkipGRU.forward`: `[batch, lookback, F]`.
- Raw output: `[batch]` logits (no sigmoid — `BCEWithLogitsLoss` is applied
  directly to logits during training).
- `model.take_prob(logits)`: `[batch]` probabilities in `[0, 1]`, sigmoid
  applied only at inference, never before the loss.

## Limitations — read this before trusting any number from this pipeline

- **No real-ticker downloader exists in `Practice/`.** Every CLI command
  currently runs on `synthetic_data.py`'s deterministic synthetic OHLC
  (clearly labeled `SYNTHETIC — NOT REAL MARKET DATA` everywhere it appears),
  selected by `config.yaml`'s `data.seed`. Wiring in a real feed (the main
  repo's `data.py` uses `yfinance`) is the natural next step, deliberately
  not done here to keep `Practice/` decoupled from that module's unrelated
  `torch`/`Model` imports.
- **`scaler.json` is not applied anywhere.** No normalization layer was ever
  added in Steps 3-8's actual GRU training path; the file is written for
  inspection only and says so explicitly (`"applied_at_inference": false`).
- **Event-driven portfolio valuation, not daily mark-to-market.** Per
  `STEP_9_PORTFOLIO_BACKTEST.md`'s own note, the equity curve only has
  checkpoints at entry/exit events, not at every calendar day — correct at
  each checkpoint, not smooth between them. Upgrading to true daily
  mark-to-market is flagged there as a deliberate, not-yet-built next step.
- **Portfolio-level `entry_price`/`stop` are reconstructed, not stored.**
  Step 5's `TradeResult` never exposes the stop price or the pre-cost raw
  entry price it used internally. `cli.py`'s `evaluate`/`signals` commands
  re-derive both from the same OHLC + ATR inputs Step 5 already used
  (`cli._attach_portfolio_columns`), without modifying `trade_simulation.py`
  itself — a small, documented seam rather than a silent assumption.
  The reconstructed `exit_price` is Step 5's cost-adjusted figure (not raw),
  so portfolio dollar accounting carries a sub-few-bps approximation versus
  a version where Step 5 exported these fields directly.
- **Risk/evaluation reporting is a subset of `STEP_9_PORTFOLIO_BACKTEST.md`'s
  full menu.** `fold_metrics.csv` reports trade counts, accuracy, expectancy,
  label precision vs. profit rate, final equity, max drawdown, and benchmark
  return for one fold. It does **not** currently include: breakdowns by
  regime or calendar year, or block-bootstrap confidence intervals on
  expectancy/Sharpe — both are specified in Step 9's doc and are a real gap
  between what's built and what the brief originally asked for.
- **Small feature set, local-optimum HMM.** `hmmlearn`'s Baum-Welch fit can
  converge to a local optimum (you'll sometimes see "Model is not
  converging" from `hmmlearn` itself — informational, not fatal); this
  project does not run multiple-restart model selection.
- **No broker execution anywhere.** This is a backtest/research pipeline
  only — `portfolio_backtest.py` never places a real order.

## What this has NOT proven

Building this pipeline correctly is necessary but not sufficient to answer
the question it was built to ask. **Nothing in this repository claims, or
has tested, that regime/seasonal features actually improve real-ticker,
out-of-sample trading results versus Variant A (no model at all).** That
comparison — run on real tickers via `ablation_runner.py`'s variants A-E,
with the Step 9 risk reporting Step 8 exists to produce — is the actual
experiment this project was built to run. It is the last mile intentionally
left undone by this README, because it is not an engineering step; it's the
result, whatever it turns out to be, including a negative one.
