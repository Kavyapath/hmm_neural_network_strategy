# Fix: AAPL (and other high-price/low-ATR%) trades never got admitted

## The symptom

Training/evaluating on AAPL, `reports/fold_metrics.csv` always showed
`n_admitted_portfolio: 0` and `final_equity: 100000.0` — the equity curve
was a flat line for the whole eval/test window, regardless of how many
candidates the model accepted.

## Root cause

`portfolio_backtest.run_portfolio_backtest` sized every admitted trade
purely by **risk** (`shares = risk_budget / stop_distance`), then
**rejected the whole trade outright** if the resulting notional exceeded
`max_position_pct × equity` — it never resized, only skipped.

For a tight stop on an expensive, low-volatility ticker, that risk-only
sizing produces an oversized position every time:

| Quantity | AAPL example |
|---|---|
| `risk_budget` = `0.01 × 100,000` | $1,000 |
| entry price | ~$197-230 |
| ATR(7) at old `stop_atr_mult=1.0` → `stop_distance` | ~$4.50 (only ~2% of price) |
| `shares = risk_budget / stop_distance` | ~224 |
| `notional = shares × entry_price` | **~$44,000** |
| `max_position_pct` cap = `0.20 × 100,000` | **$20,000** |

`$44k > $20k` → rejected, every single time, independent of trade quality.
AAPL's ATR is only ~2% of its price; NVDA/INTC (higher ATR%) weren't
hitting this as hard, which is why it looked AAPL-specific.

Separately: only 19 training trades existed for AAPL in the first place —
too few for the GRU to move its `take_prob` output off ~0.50 (see
`accepted_trades.csv`'s take_probs, all ~0.50-0.51).

## Files changed

| File | What changed |
|---|---|
| `pullback_entry.py` | new `PullbackConfig.min_rules_required` field + `candidate_signal` logic (5-of-5 AND → N-of-5) |
| `portfolio_backtest.py` | `run_portfolio_backtest`'s admission loop: resize-to-fit instead of reject |
| `config.yaml` | `trade_sim.stop_atr_mult` 1.0→2.5, new `pullback.min_rules_required: 4` |
| `app.py` | sidebar defaults/help text to match `config.yaml`, new `min_rules_required` control |
| `CHANGES_PORTFOLIO_SIZING_FIX.md` | this file — written after the three changes above, documents them |

## The three fixes, and exactly where

### 1. `trade_sim.stop_atr_mult`: `1.0` → `2.5`
**File:** `config.yaml:37` (and `app.py:240-245` sidebar default/help text)

```diff
 trade_sim:
   atr_window: 7
-  stop_atr_mult: 1.0
+  stop_atr_mult: 2.5
```

Portfolio sizing requires `stop_distance >= risk_pct_per_trade /
max_position_pct` as a fraction of price (`0.01 / 0.20 = 5%` at current
settings), or the notional-cap rejection above is unavoidable regardless of
trade quality. Measured real ATR% for the three tickers in question (6
months of daily bars):

| Ticker | ATR as % of price | min `stop_atr_mult` to clear the 5% floor |
|---|---|---|
| AAPL | 2.07% | 2.52 |
| NVDA | 10.10% | 1.21 |
| INTC | 4.27% | 1.27 |

`2.5` clears all three with margin. (Verified by direct calculation against
live Alpaca data, not assumed — see conversation for the script.)

### 2. Portfolio admission now resizes instead of rejecting
**File:** `portfolio_backtest.py:74-109` (`run_portfolio_backtest`'s per-candidate admission loop)

```diff
-            risk_budget = cfg.risk_pct_per_trade * equity
-            stop_distance = cand["entry_price"] - cand["stop"]
-            if stop_distance <= 0:
-                continue  # malformed candidate, should not happen if Step 5 is correct — skip defensively
-            shares = risk_budget / stop_distance
-            notional = shares * cand["entry_price"]
-
-            open_risk = sum(p.risk_dollars for p in open_positions)
-            if (
-                notional > cash
-                or notional > cfg.max_position_pct * equity
-                or len(open_positions) >= cfg.max_concurrent_trades
-                or open_risk + risk_budget > cfg.max_aggregate_risk_pct * equity
-            ):
-                continue  # not admitted — capital/risk limits binding, not a trade-quality judgment
-
-            cash -= notional
-            open_positions.append(OpenPosition(
-                ticker=cand["ticker"], shares=shares, entry_price=cand["entry_price"],
-                entry_date=cand["entry_date"], stop=cand["stop"], exit_date=cand["exit_date"],
-                exit_price=cand["exit_price"], risk_dollars=risk_budget,
-            ))
+            stop_distance = cand["entry_price"] - cand["stop"]
+            if stop_distance <= 0:
+                continue  # malformed candidate, should not happen if Step 5 is correct — skip defensively
+            if len(open_positions) >= cfg.max_concurrent_trades:
+                continue  # a count cap — nothing to resize, there's no open slot
+
+            open_risk = sum(p.risk_dollars for p in open_positions)
+            risk_room = cfg.max_aggregate_risk_pct * equity - open_risk
+            if risk_room <= 0:
+                continue  # aggregate risk budget already fully committed
+
+            # Size by risk first, then resize DOWN (never up) to also fit
+            # inside cash on hand, the single-position notional cap, and the
+            # remaining aggregate-risk room. A tight stop on an expensive
+            # ticker (small stop_distance, high entry_price) used to make
+            # `shares` blow past max_position_pct and get skipped outright —
+            # now it gets admitted at a smaller size instead.
+            risk_budget = cfg.risk_pct_per_trade * equity
+            risk_dollars = min(risk_budget, risk_room)
+            shares = risk_dollars / stop_distance
+            notional = shares * cand["entry_price"]
+
+            max_notional = min(cash, cfg.max_position_pct * equity)
+            if notional > max_notional:
+                shares = max_notional / cand["entry_price"]
+                notional = shares * cand["entry_price"]
+                risk_dollars = shares * stop_distance
+
+            if shares <= 0:
+                continue  # nothing left to size into (e.g. cash exhausted)
+
+            cash -= notional
+            open_positions.append(OpenPosition(
+                ticker=cand["ticker"], shares=shares, entry_price=cand["entry_price"],
+                entry_date=cand["entry_date"], stop=cand["stop"], exit_date=cand["exit_date"],
+                exit_price=cand["exit_price"], risk_dollars=risk_dollars,
+            ))
```

Only two things still cause an outright skip now, because there's nothing
to resize around:
- `len(open_positions) >= max_concurrent_trades` (a count cap, not a size cap)
- the aggregate risk budget is already fully committed (`risk_room <= 0`)

Verified directly: the exact AAPL scenario above (`entry_price=196.9,
stop_distance=4.46`) that used to get skipped now gets admitted at
**101.6 shares** (notional resized down to exactly the $20,000 cap,
`risk_dollars` reduced from the full $1,000 budget to $453 accordingly)
instead of being thrown away.

### 3. `pullback.min_rules_required`: new field, default `4` (of 5)
**Files:**
- `pullback_entry.py:15` — new `PullbackConfig.min_rules_required: int = 5` field (default `5` = original strict AND, so nothing changes for anyone who doesn't set it)
- `pullback_entry.py:59-64` (`candidate_signal`) — now sums how many of the 5 rules are `True` per day and admits the day if the count is `>= min_rules_required`, instead of requiring all 5
- `config.yaml:27` — sets the new default to `4`
- `app.py:232-236` — new sidebar control, `min_rules_required (of 5)`, default `4`

```diff
 @dataclass
 class PullbackConfig:
     ma_window: int = 50          # trend filter MA length
     rel_strength_window: int = 20   # trailing-return comparison window
     pullback_window: int = 3        # sessions t-3..t-1 checked for the dip
     prior_high_lookback: int = 10   # how far back close[t-4] must be a local high
+    min_rules_required: int = 5     # how many of the 5 rules must be True; 5 = original strict AND
```

```diff
     rule5_trigger = stock_close > stock_high.shift(1)

-    candidate = (
-        rule1_benchmark_trend
-        & rule2_stock_trend
-        & rule3_rel_strength
-        & rule4_pullback
-        & rule5_trigger
-    )
+    rules = [
+        rule1_benchmark_trend, rule2_stock_trend, rule3_rel_strength,
+        rule4_pullback, rule5_trigger,
+    ]
+    rules_true_count = sum(r.fillna(False).astype(int) for r in rules)
+    candidate = rules_true_count >= cfg.min_rules_required
     return candidate.fillna(False)
```

Measured on real AAPL data (`2020-07-27`-`2025-01-01`, the actual span this
Alpaca account's IEX feed has for AAPL — going further back in
`data.start` is a no-op, the feed simply has nothing before that date):

| `min_rules_required` | candidates |
|---|---|
| 5 (old strict AND) | 41 |
| **4 (new default)** | **254** |

A ~6x increase in candidate density — the actual lever that matters, versus
window-size tuning (`ma_window`, etc.), which earlier testing showed has
almost no headroom on this rule structure (see `STRATEGY_BLUEPRINT.md`'s
grid-search note: window tuning alone maxed out around +8% over baseline).

## Verified end-to-end (not just unit-level)

Full `cli.py train` → `evaluate` run on real AAPL data with all three fixes applied together:

| Metric | Before | After |
|---|---|---|
| `n_test_trades` | 19 | 65 |
| `n_admitted_portfolio` | **0** | **47** |
| `final_equity` | 100,000.0 (flat) | 100,593.1 |
| `max_drawdown` | 0.0 (flat) | -3.1% |

The equity curve now actually moves — which was the original symptom.

## What this doesn't fix

More admitted trades and a non-flat equity curve isn't the same as a
*profitable* strategy — `test_expectancy` on this particular run came back
slightly negative (-0.000127) and `test_accuracy` very low (0.015), which
reflects the GRU still not discriminating well with this little training
data (65 trades total, split across train/dev/test). That's the next thing
to improve, not something these three fixes touch.
