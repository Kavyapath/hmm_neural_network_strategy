# Practice/ — Module Map

What each file does today, how data flows through the pipeline, and exactly
where the GARCH volatility-regime module slots in to replace `seasonal_features.py`.

## Data flow (left to right)

```
real_data.py / synthetic_data.py
        │  OHLC per ticker + benchmark
        ▼
pullback_entry.py            → boolean candidate-signal days (5 rule checks)
        │
        ▼
trade_simulation.py          → resolves each candidate into a labeled trade
        │                        (entry/stop/target/timeout, net_return, label)
        ▼
feature_blocks.py            → builds the per-day feature frame from whichever
        │                        blocks the active variant uses (technical is
        │                        always on; regime/seasonal are switchable)
        │        ┌── technical_features.py
        │        ├── regime_features.py  (needs causal_hmm.py, frozen pre-fold)
        │        └── seasonal_features.py   ← REPLACE WITH volatility_features.py
        ▼
gru_classifier.py / logreg_baseline.py   → take/skip probability per candidate
        ▼
walk_forward.py               → expanding-window folds, dev-set threshold pick,
        │                        purges overlapping trades
        ▼
portfolio_backtest.py         → capital accounting, equity curve, drawdown
        ▼
ablation_runner.py / cli.py / app.py    → orchestration, CLI commands, Streamlit UI
```

## Existing modules

| File | Role |
|---|---|
| `real_data.py` | Pulls real OHLC via Alpaca's market-data API (`data.alpaca.markets`). Credentials: env vars first, then `~/.secrets/alpaca_paper.env`. |
| `synthetic_data.py` | Deterministic synthetic OHLC generator (HMM-driven, fixed seed) for smoke tests only — not a market model. |
| `causal_hmm.py` | `CausalHMM`: Gaussian-emission HMM with causal (forward-only) filtering. Fit once per fold on train-only returns, then frozen. |
| `regime_features.py` | `fit_regime_model()` fits `CausalHMM` on train-window returns only; `build_regime_features()` runs the frozen model causally across the full series → 8 columns (`regime_bear/mid/bull`, `regime_next_*`, `regime_entropy`, `regime_persistence`). |
| `seasonal_features.py` | Cyclic day-of-week / month-of-year sin/cos encodings. Deterministic from the calendar, zero fit/leakage risk. **← being replaced.** |
| `technical_features.py` | Always-on indicator block: `rsi14`, `mom20`, `mom60`, `vol20`, `macd`, `dist_ma50`. All causal (use only `close[:t]`). |
| `feature_blocks.py` | `VARIANT_BLOCKS` dict mapping ablation variant name → which of `{technical, regime, seasonal}` it uses. `build_feature_frame()` dispatches to the right builder(s) and concats. |
| `pullback_entry.py` | `candidate_signal()`: 5-rule pullback-in-an-uptrend entry filter (benchmark trend, stock trend, relative strength, pullback-off-a-high, breakout trigger). Produces candidate signal days, not final trades. |
| `trade_simulation.py` | `simulate_trade()` / `build_labeled_dataset()`: turns each candidate day into a resolved trade (ATR-based stop/target, max holding period, commission/slippage), with a 1/0 take/skip label. |
| `gru_classifier.py` | `TakeSkipGRU` + `CandidateWindowDataset` + `train_take_skip_model()`: binary take/skip classifier over a lookback window of the feature frame, one logit per candidate trade. |
| `logreg_baseline.py` | Variant-B baseline: same windows as the GRU, but only the last timestep (signal-day state) fed to `sklearn.LogisticRegression`. |
| `walk_forward.py` | `Fold` dataclass + `run_fold()`: expanding train/dev/test windows, dev-set threshold grid search (`select_threshold`), purges overlapping trades at fold boundaries. |
| `portfolio_backtest.py` | Capital accounting over already-simulated trades: admission control (max concurrent, max aggregate risk, max position size), equity curve, drawdown, from real cash flows (not summed % returns). |
| `ablation_runner.py` | Runs variants A–E (`A_rule_based`, `B_logreg`, `C_gru_no_regime`, `D_gru_full`, `E_gru_no_seasonal`) through the same fold machinery so they're directly comparable; only the feature frame + fit/predict step differs per variant. |
| `cli.py` | CLI entry point: `preprocess`, `train`, `evaluate`, `evaluate-holdout`, `signals`. Wires every module above into one runnable pipeline, writes `runs/<run_id>/`. |
| `app.py` | Streamlit control panel over `cli.py` (subprocess calls) — ticker/date/mode pickers, train/eval(dev)/test(holdout) tabs, report rendering, Alpaca live-quote tab. |
| `test_*.py` | Unit/smoke tests for `ablation_runner`, `portfolio_backtest`, `walk_forward`, and an end-to-end synthetic smoke test. |

## New module: `volatility_features.py`

Replaces `seasonal_features.py` in the feature-block rotation. Forecasts
**magnitude** of movement (GARCH), not direction — classifies each day into
a volatile/neutral regime relative to the ticker's own trailing history.

**Where it plugs in:**

1. **New file** — `Practice/volatility_features.py`, same shape/signature
   convention as `seasonal_features.py`:
   ```python
   def build_volatility_features(close: pd.Series, garch_model=None) -> pd.DataFrame:
       ...  # returns columns: garch_vol_fcst, vol_percentile_1y, vol_regime_flag
   ```
   Must be **walk-forward fit** like `causal_hmm.py` — refit GARCH(1,1) params
   every ~21 trading days on an expanding, train-only window; never fit once
   on the full series. Needs a `fit_volatility_model(close, train_end)` sibling
   function mirroring `regime_features.fit_regime_model()`'s pattern (frozen
   parameters handed into the per-fold feature build, not refit inside it).

2. **`feature_blocks.py`** — swap the import and variant membership:
   ```python
   from volatility_features import build_volatility_features   # was seasonal_features

   VARIANT_BLOCKS = {
       "A_rule_based": frozenset(),
       "B_logreg": frozenset({"technical", "regime", "volatility"}),
       "C_gru_no_regime": frozenset({"technical", "volatility"}),
       "D_gru_full": frozenset({"technical", "regime", "volatility"}),
       "E_gru_no_seasonal": frozenset({"technical", "regime"}),  # rename if desired
   }
   ```
   and add a `"volatility"` branch in `build_feature_frame()` alongside the
   existing `if "regime" in blocks:` block — it will need the fold-frozen
   GARCH model passed in the same way `regime_model` is, since GARCH also
   has a fit step that must not leak future data.

3. **`walk_forward.py`** — wherever `fit_regime_model()` is called once per
   fold (train-only) and handed to `build_feature_frame`, add the matching
   `fit_volatility_model()` call right next to it, same lifecycle.

4. **Optional, beyond a feature column** — two other integration points if
   you want volatility to do more than inform the GRU:
   - **Gate** in `pullback_entry.py` or `walk_forward.py`: force-reject any
     candidate generated while `vol_regime_flag == "volatile"`, independent
     of what the model predicts.
   - **Position sizing** in `portfolio_backtest.py`: scale
     `risk_pct_per_trade` by `target_vol / garch_vol_fcst` (capped, e.g.
     0.25x–2x) instead of using the flag as a binary filter.

5. **`ablation_runner.py` / `cli.py` / `app.py`** — no structural changes
   needed; they already treat the feature-block set as data (`VARIANT_BLOCKS`
   keys, `config.yaml`'s `ablation.feature_blocks` list). Once `"volatility"`
   exists as a valid block name, it's selectable from the existing UI/config
   the same way `"seasonal"` was.
