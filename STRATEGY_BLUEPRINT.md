# Strategy Blueprint — end-to-end walkthrough

One document that follows a single trade candidate through every module in
`Practice/`, from raw OHLC to a portfolio-level equity curve, and says
exactly which function does which job and who actually calls it in
production (`cli.py`) versus research tooling (`ablation_runner.py`,
`app.py`).

**The worked example uses made-up round numbers for one hypothetical
ticker ("XYZ") on one hypothetical date.** It is for tracing data shapes
and formulas only — not a real trade, not a backtest result, not evidence
of anything. Treat every number below the way `synthetic_data.py` treats
its own output: illustrative scaffolding.

---

## 1. The pipeline, top to bottom

```
raw OHLC (synthetic_data.py or real_data.py)
        │
        ▼
candidate_signal()                    pullback_entry.py      "is today a valid entry day?"
        │  (boolean Series)
        ▼
build_labeled_dataset()               trade_simulation.py    "what actually happened if I took it?"
        │  (one row per RESOLVED trade: net_return, label, exit_date, ...)
        ▼
purge_overlapping() × fold boundary   walk_forward.py         "drop trades whose outcome leaks past the split"
        │
        ├──────────────┬──────────────┬───────────────────┐
        ▼              ▼              ▼                   │
technical_features   regime_features  volatility_features  │  all three are CAUSAL — row t uses
  (always on)         (fit_regime_     (fit_volatility_     │  only data up to t, frozen at fit time
                        model, Step 3)   model, Step 9)      │
        └──────────────┴──────────────┴───────────────────┘
                         │  concat (feature_blocks.build_feature_frame)
                         ▼
              CandidateWindowDataset                        gru_classifier.py   "last `lookback` rows
                         │  (window [lookback, n_features], label)                ending at signal_date"
                         ▼
              train_take_skip_model() / TakeSkipGRU          gru_classifier.py   "learn take vs. skip"
                         │  (take_prob per candidate, dev-set only for selection)
                         ▼
              select_threshold()                             walk_forward.py     "pick the take_prob cutoff
                         │                                                        that maximizes DEV expectancy"
                         ▼
              accepted trades (take_prob >= threshold)
                         │
                         ▼
              run_portfolio_backtest()                       portfolio_backtest.py  "capital accounting:
                         │  (equity_curve, admitted trades)                          sizing, caps, cash flow"
                         ▼
                  equity curve / drawdown / calibration PNGs, fold_metrics.csv
```

Everything above the "concat" line is **per-ticker, independent**. Everything
from `CandidateWindowDataset` down is **pooled across every ticker in the
run** — one GRU, one threshold, one portfolio.

---

## 2. Stage-by-stage, with the worked example

### 2.1 Price data in

- **Function:** `generate_synthetic_universe()` (`synthetic_data.py`) or
  `load_real_universe()` (`real_data.py`), chosen by `cfg["data"]["source"]`
  in `cli._load_price_data`.
- **Shape:** `price_data: dict[ticker -> OHLC DataFrame]`, `benchmark_close: Series`. Both on the same business-day calendar.

**Example:** `price_data["XYZ"]` has columns `open, high, low, close`,
indexed by date. On `2024-03-01`, `close = 118.80`.

### 2.2 Candidate signal — "is today an entry day?"

- **Function:** `candidate_signal(stock_close, benchmark_close, stock_high, cfg)` (`pullback_entry.py`).
- Five rules, all boolean, all causal (only `.shift()`/`.rolling()` on past bars):

| Rule | Condition | Example value on 2024-03-01 |
|---|---|---|
| 1. Benchmark trend | `benchmark_close > benchmark_close.rolling(50).mean()` | SPY 512 > MA50 505 → **True** |
| 2. Stock trend | `close > close.rolling(50).mean()` | XYZ 118.80 > MA50 110.20 → **True** |
| 3. Relative strength | `stock_ret20 > bench_ret20` | XYZ +9.1% (20d) > SPY +3.4% → **True** |
| 4. Pullback after a prior high | prior high at `t-4`, dip somewhere in `t-3..t-1` | high=121.50 at t-4, dip to 117.90 at t-2 → **True** |
| 5. Trigger | `close[t] > high[t-1]` | 118.80 > 117.95 → **True** |

All five True → `candidate_signal` is `True` on `2024-03-01` for XYZ.

### 2.3 Trade simulation — "what happens if I take it?"

- **Functions:** `average_true_range()`, `simulate_trade()` (one candidate),
  `build_labeled_dataset()` (all candidates for a ticker) — all in `trade_simulation.py`.
- Entry is **next session's open** (no same-day fill); stop/target are fixed
  at entry from ATR; costs applied via `_apply_costs` (slippage + commission, bps).

**Example** (`cfg.atr_window=14, stop_atr_mult=1.5, target_r_mult=2.0, max_holding_sessions=7`):

| Quantity | Value |
|---|---|
| `signal_date` | 2024-03-01 |
| `entry_date` | 2024-03-04 (next session) |
| `atr_at_signal` | 4.35 |
| raw entry (next open) | 120.00 |
| `entry_price` (after +2bps slip, +1bps comm) | 120.04 |
| `stop` = `120.00 - 1.5×4.35` | 113.47 |
| `stop_distance` | 6.53 |
| `target` = `120.00 + 2.0×6.53` | 133.06 |
| Session 3 high | 133.40 → **target hit** |
| `exit_price` (raw 133.06, after costs) | 133.02 |
| `net_return` = `133.02/120.04 - 1` | **+10.8%** |
| `r_multiple` | **+2.0** |
| `label` (`reason=="target" and net_return>0`) | **1** |

This row — tagged `ticker="XYZ"` — is one row of `build_labeled_dataset`'s
output DataFrame. Unresolved trades (not enough future bars, or a stop/ATR
`NaN`) are **dropped**, never labeled 0.

### 2.4 Fold split + purge — no leakage across train/dev/test

- **Functions:** `Fold` (dataclass), `purge_overlapping(labeled_trades, cutoff)` (`walk_forward.py`).
- A trade whose `exit_date` falls *after* the fold boundary used price action
  the model isn't allowed to see yet at that boundary — purged from that side.

**Example:** fold `train_end = 2024-02-20`. The XYZ trade above has
`signal_date = 2024-03-01`, so it's a **dev/test**-side trade, not train —
it's placed by `signal_date`, then purged from whichever side its
`exit_date` (2024-03-07) would leak past.

### 2.5 Feature blocks — technical (always on) + regime + volatility

Three independently-switchable blocks, concatenated by
`feature_blocks.build_feature_frame(close, blocks, regime_model=, volatility_model=)`.
`blocks` comes from `config.yaml`'s `ablation.feature_blocks` — e.g.
`["technical", "regime", "volatility"]`.

**a) Technical** — `build_technical_features(close)` (`technical_features.py`), 6 columns, always included:

| `rsi14` | `mom20` | `mom60` | `vol20` | `macd` | `dist_ma50` |
|---|---|---|---|---|---|
| 61.2 | 0.071 | 0.142 | 0.018 | 0.011 | 0.077 |

**b) Regime** — fit once per fold (`fit_regime_model(close, train_end, n_states=3)`,
`regime_features.py`), then run causally across the whole series
(`build_regime_features`), 8 columns:

| `regime_bear` | `regime_mid` | `regime_bull` | `regime_next_bear` | `regime_next_mid` | `regime_next_bull` | `regime_entropy` | `regime_persistence` |
|---|---|---|---|---|---|---|---|
| 0.04 | 0.21 | **0.75** | 0.05 | 0.19 | 0.76 | 0.41 | 0.97 |

Under the hood: `CausalHMM.filter_step` runs the scaled forward algorithm
one observation at a time — `filtered_t` only depends on `obs_t` and the
previous internal state, never on anything after `t` (`causal_hmm.py`).

**c) Volatility** — fit once per fold (`fit_volatility_model(close, train_end)`,
`volatility_features.py`), then forecast causally (`build_volatility_features`), 3 columns:

| `garch_vol_fcst` (annualized) | `vol_percentile_1y` | `vol_regime_flag` |
|---|---|---|
| 0.34 (34%) | 0.82 | **1.0** (top-quartile vol) |

Under the hood: `_forecast_sigma2_series` runs `sigma2[t] = omega + alpha*r[t-1]^2 + beta*sigma2[t-1]`
with `omega/alpha/beta` **frozen** at `fit_volatility_model` time — `sigma2[t]`
never sees `r[t]` itself, only `r[t-1]` and earlier. `ljung_box_test` runs
inside `fit_garch` before the MLE fit, and warns if the fit window's returns
still have significant serial autocorrelation the plain demeaned mean
equation isn't accounting for.

**Concatenated row for XYZ on 2024-03-01** (17 columns: 6 + 8 + 3):

```
rsi14=61.2  mom20=0.071 ... regime_bull=0.75 ... garch_vol_fcst=0.34 vol_regime_flag=1.0
```

### 2.6 Windowing — `CandidateWindowDataset`

- Takes the labeled-trades DataFrame (one row per resolved trade, tagged by
  `ticker`) + `feature_frames: dict[ticker -> DataFrame]`, slices the
  **`lookback` rows ending at `signal_date`** out of that ticker's feature
  frame, per candidate.
- **Example:** `lookback=30` → the window for the XYZ candidate is the
  17-column feature frame's rows `[signal_date-29 .. signal_date]`, shape
  `(30, 17)`. Candidates with fewer than `lookback` prior rows (not enough
  warm-up history) are silently dropped — never padded with zeros.

### 2.7 Model — `TakeSkipGRU`

- `train_take_skip_model(train_ds, dev_ds, cfg, device)` (`gru_classifier.py`):
  one GRU layer → last hidden state → one linear head → one logit per
  candidate. Trained with `BCEWithLogitsLoss` (class-imbalance corrected via
  `estimate_pos_weight`, computed from **train only**), early-stopped on dev
  loss (`cfg.patience`).
- **Inference:** `model(window)` → logit → `model.take_prob(logit)` =
  `sigmoid(logit)`.

**Example:** forward pass on the XYZ window → logit `0.92` → `take_prob = 0.72`.

### 2.8 Threshold selection — dev only, never test

- `select_threshold(dev_take_probs, dev_net_returns, min_trades, grid)` (`walk_forward.py`):
  scans a grid of thresholds, picks the one with the **highest dev-set mean
  net_return** among thresholds that accept at least `min_trades` dev trades.

**Example:** grid scan on dev picks `threshold = 0.65` (clears
`min_trades_for_threshold=20`, highest dev expectancy among those that do).
XYZ's `take_prob=0.72 >= 0.65` → **trade accepted** into the test-side
portfolio input.

### 2.9 Portfolio backtest — capital accounting

- `run_portfolio_backtest(accepted_candidates, cfg)` (`portfolio_backtest.py`):
  ranks same-day candidates deterministically (`take_prob` desc), then walks
  every entry/exit event in date order, sizing each admitted trade by
  **risk budget ÷ stop distance**, subject to a single-position cap, a
  max-concurrent-trades cap, and an aggregate open-risk cap.

**Example** (`starting_cash=100,000`, `risk_pct_per_trade=0.01`, `max_position_pct=0.20`):

| Quantity | Value |
|---|---|
| `risk_budget` = `0.01 × 100,000` | 1,000 |
| `stop_distance` (from §2.3) | 6.53 |
| `shares` = `1,000 / 6.53` | 153.1 |
| `notional` = `153.1 × 120.04` | 18,378 |
| cap check: `18,378 < 0.20 × 100,000 = 20,000` | **passes → admitted** |
| on exit at 133.02: proceeds = `153.1 × 133.02` | 20,369 |
| equity after this trade closes | `100,000 − 18,378 + 20,369 = 101,991` |

That new equity number becomes one point on `equity_curve`, written to
`reports/fold_metrics.csv` and plotted in `equity_curve.png`.

---

## 3. What actually runs in production — call graph by `cli.py` command

`cli.py` is the only production entry point (`app.py` is a Streamlit
wrapper that shells out to it). `ablation_runner.py`'s own functions
(`evaluate_variant`, `run_all_variants`, `run_config_search`,
`benchmark_buy_and_hold`) are **research-only** — nothing in `cli.py` calls
them; `cli.py` only reuses its `ExperimentLog`/`log_experiment` logging
helpers for its own run log.

### `python cli.py preprocess --config config.yaml --out <dir>`
```
cmd_preprocess
 ├─ load_config
 ├─ _load_price_data → generate_synthetic_universe | load_real_universe
 └─ per ticker: candidate_signal → build_labeled_dataset
                                    (→ simulate_trade per candidate, average_true_range)
 writes: candidates.parquet
```

### `python cli.py train --config config.yaml --run-id <id>`
```
cmd_train
 ├─ load_config, _resolve_device, _load_price_data
 ├─ _single_fold  (or an explicit Fold from --train-start/... args)
 ├─ _build_feature_frames
 │   └─ per ticker: fit_regime_model (if "regime" in blocks)
 │                   fit_volatility_model (if "volatility" in blocks)
 │                   build_feature_frame → build_technical_features
 │                                          build_regime_features
 │                                          build_volatility_features
 ├─ _labeled_trades_by_fold
 │   └─ per ticker: candidate_signal → build_labeled_dataset → purge_overlapping ×2
 ├─ [insufficient-data guard: too few train/dev trades → write run_metadata.json, stop]
 ├─ CandidateWindowDataset ×2 (train, dev)
 ├─ train_take_skip_model → TakeSkipGRU, estimate_pos_weight
 ├─ _score (dev) → select_threshold
 ├─ [insufficient-data guard: no threshold clears min_trades → write run_metadata.json, stop]
 ├─ torch.save(model.state_dict())                  → model.pt
 ├─ _hmm_to_json                                     → regime_model.json
 ├─ _garch_to_json                                   → volatility_model.json
 ├─ plot_heteroscedasticity (per ticker with a volatility_model)
 │                                                    → heteroscedasticity_<ticker>.png
 └─ log_experiment                                   → experiment_log.jsonl, run_metadata.json,
                                                         feature_schema.json, scaler.json, history.json
```

### `python cli.py evaluate --run-id <id>`
```
cmd_evaluate
 ├─ load_config, run_metadata.json  [stop if insufficient_data]
 ├─ _load_price_data
 ├─ _hmm_from_json, _garch_from_json  (per ticker, from the saved run)
 ├─ build_feature_frame (frozen regime/volatility models, same as train)
 ├─ _labeled_trades_by_fold → CandidateWindowDataset (test slice)
 ├─ TakeSkipGRU + load_state_dict(model.pt) → _score
 ├─ accepted = test_probs >= saved threshold
 ├─ _attach_portfolio_columns (re-derives entry_price/stop from average_true_range)
 ├─ run_portfolio_backtest
 └─ writes: reports/fold_metrics.csv, accepted_trades.csv,
            equity_curve.png, drawdown.png, calibration.png
```

### `python cli.py evaluate-holdout --run-id <id> --buffer-start ... --test-start ... --test-end ...`
Same shape as `evaluate`, but pulls **fresh price data** up to `--test-end`
instead of reusing the original fold — the regime/volatility models stay
the ones frozen at the original `train_end` (no refit), so running them
forward over new dates is causal inference, not a leak. Writes to
`reports_holdout_<start>_<end>/` instead of `reports/`.

### `python cli.py signals --run-id <id> --as-of <date>`
```
cmd_signals
 ├─ load_config, run_metadata.json  [stop if insufficient_data]
 ├─ _load_price_data, _hmm_from_json, _garch_from_json
 ├─ per ticker: build_feature_frame → candidate_signal
 │               for each fired signal in the last 10 sessions up to --as-of:
 │                 slice the lookback window → model(window) → take_prob
 └─ prints: ticker, date, take_prob, threshold, would_take — no files written
```

### `streamlit run app.py`
Pure UI: builds a `config.yaml`-shaped dict from the sidebar form
(`build_config`), writes it to `configs/<run_id>.yaml`, and shells out to
`cli.py train|evaluate|evaluate-holdout` as a subprocess (`run_cli`),
rendering back `history.json`, `heteroscedasticity_*.png`, and the
`reports*/`'s CSVs/PNGs. Also has a standalone "Live quote" tab that calls
Alpaca's market-data API directly (`alpaca_latest_trades`) — unrelated to
the train/evaluate pipeline.

---

## 4. Research-only tooling (not part of the production path)

`ablation_runner.py` runs the **Step 8 baselines/ablation study** — five
variants (`A_rule_based`, `B_logreg`, `C_gru_no_regime`, `D_gru_full`,
`E_gru_no_volatility`) over the same folds, to answer "does the regime
block / volatility block / GRU over logreg actually help, or is the rule
alone just as good?" It reuses `walk_forward`'s `Fold`, `purge_overlapping`,
`select_threshold` so every variant's split/purge/threshold logic is
identical — only the feature frame and the fit/predict step change per
variant. `run_config_search` does a small `hidden_size × lr` grid search on
`D_gru_full` only, selected by **dev** expectancy, never test. Nothing in
`cli.py` or `app.py` calls into this file's experiment functions — it's a
separate script you run to decide which variant to ship, not something the
trained/served model depends on at runtime.

---

## 5. Artifact map (what `cli.py train` + `evaluate` leave on disk)

```
runs/<run_id>/
├── config_used.yaml          the exact config this run trained with
├── model.pt                  TakeSkipGRU state_dict
├── history.json              per-epoch train_loss/val_loss/val_acc
├── regime_model.json         one frozen CausalHMM per ticker
├── volatility_model.json     one frozen GarchParams per ticker
├── heteroscedasticity_<ticker>.png   returns vs. vol forecast, train|dev|test marked
├── feature_schema.json       exact ordered column names + lookback
├── scaler.json               informational only — not applied anywhere
├── run_metadata.json         fold, threshold, dev_expectancy, insufficient_data flag
├── experiment_log.jsonl      append-only log, one line per cli_train/evaluate call
└── reports/  (or reports_holdout_<start>_<end>/)
    ├── fold_metrics.csv
    ├── accepted_trades.csv
    ├── equity_curve.png
    ├── drawdown.png
    └── calibration.png
```

---

## 6. Config knobs → where they're consumed

| `config.yaml` key | Consumed by |
|---|---|
| `data.*` | `_load_price_data` → `generate_synthetic_universe` / `load_real_universe` |
| `regime.n_states` | `fit_regime_model` |
| `pullback.*` | `PullbackConfig` → `candidate_signal` |
| `trade_sim.*` | `TradeSimConfig` → `simulate_trade` / `average_true_range` |
| `model.*` | `GRUConfig` → `TakeSkipGRU` / `train_take_skip_model` |
| `walk_forward.min_trades_for_threshold`, `.threshold_grid` | `select_threshold` |
| `portfolio.*` | `PortfolioConfig` → `run_portfolio_backtest` |
| `ablation.feature_blocks` | `build_feature_frame`'s `blocks` argument (`cli.py` and `ablation_runner.py` both read this key; it has nothing to do with the Step 8 variant table — it's just which blocks this one run turns on) |
