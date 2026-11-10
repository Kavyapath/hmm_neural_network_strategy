"""Interactive control panel for Practice/cli.py's train -> evaluate ->
evaluate-holdout pipeline.

Lets you pick the ticker list, train/eval(dev) date ranges, a separate
holdout test date range, the feature-block "model mode" (technical / regime /
volatility, any combination), and every other config.yaml knob, then runs the
existing CLI as a subprocess and renders back: the training loss/accuracy
curve, the eval(dev) report (accuracy, expectancy, equity curve, drawdown,
calibration, best trades), and the same for the 2025-2026-style holdout test.

Also has a "Live quote" tab that pulls a real-time quote from Alpaca's market
data API, using the same ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY env vars
AIStrategy/live_signal/alpaca_adapter.py already expects (paper account keys
work for market data too - it's a different host, not a different key).

Run with:
    .venv/Scripts/streamlit.exe run Practice/app.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st
import yaml

APP_DIR = Path(__file__).parent
RUNS_DIR = APP_DIR / "runs"
CONFIGS_DIR = APP_DIR / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)

RULE_LABELS = {
    "rule1_benchmark_trend": "Rule 1 — benchmark above its moving average (market uptrend)",
    "rule2_stock_trend": "Rule 2 — stock above its own moving average (stock uptrend)",
    "rule3_rel_strength": "Rule 3 — stock's trailing return beats the benchmark's (relative strength)",
    "rule4_pullback": "Rule 4 — price pulled back from a recent high",
    "rule5_trigger": "Rule 5 — today's close broke back above yesterday's high (trigger)",
}

MODE_TO_BLOCKS = {
    "Full (technical + regime + volatility)": ["technical", "regime", "volatility"],
    "No regime (technical + volatility)": ["technical", "volatility"],
    "No volatility (technical + regime)": ["technical", "regime"],
    "Technical only": ["technical"],
}

st.set_page_config(page_title="Strategy Lab", layout="wide")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def run_cli(args: list[str]) -> tuple[int, str]:
    """Runs `python cli.py <args>` from Practice/, streaming output into a
    scrolling log box, and returns (exit_code, full_output)."""
    cmd = [sys.executable, "cli.py", *args]
    log_box = st.empty()
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    for line in proc.stdout:
        lines.append(line.rstrip())
        log_box.code("\n".join(lines[-40:]))
    proc.wait()
    return proc.returncode, "\n".join(lines)


def build_config(form: dict) -> dict:
    return {
        "data": {
            "source": "alpaca",
            "tickers": form["tickers"],
            "benchmark": form["benchmark"],
            "start": form["buffer_start"].isoformat(),
            "end": form["dev_end"].isoformat(),
        },
        "regime": {"n_states": form["n_states"]},
        "pullback": {
            "ma_window": form["ma_window"],
            "rel_strength_window": form["rel_strength_window"],
            "pullback_window": form["pullback_window"],
            "prior_high_lookback": form["prior_high_lookback"],
            "min_rules_required": form["min_rules_required"],
        },
        "trade_sim": {
            "atr_window": form["atr_window"],
            "stop_atr_mult": form["stop_atr_mult"],
            "target_r_mult": form["target_r_mult"],
            "max_holding_sessions": form["max_holding_sessions"],
            "commission_bps": form["commission_bps"],
            "slippage_bps": form["slippage_bps"],
        },
        "model": {
            "lookback": form["lookback"],
            "hidden_size": form["hidden_size"],
            "dropout": form["dropout"],
            "lr": form["lr"],
            "weight_decay": form["weight_decay"],
            "grad_clip": form["grad_clip"],
            "max_epochs": form["max_epochs"],
            "patience": form["patience"],
            "seed": form["seed"],
        },
        "walk_forward": {
            "min_trades_for_threshold": form["min_trades_for_threshold"],
            "threshold_grid": [form["threshold_lo"], form["threshold_hi"], form["threshold_n"]],
        },
        "portfolio": {
            "starting_cash": form["starting_cash"],
            "risk_pct_per_trade": form["risk_pct_per_trade"],
            "max_position_pct": form["max_position_pct"],
            "max_concurrent_trades": form["max_concurrent_trades"],
            "max_aggregate_risk_pct": form["max_aggregate_risk_pct"],
        },
        "ablation": {"feature_blocks": MODE_TO_BLOCKS[form["mode"]]},
    }


def load_json(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def load_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def render_report(run_dir: Path, reports_subdir: str, label: str) -> None:
    reports_dir = run_dir / reports_subdir
    metrics = load_csv(reports_dir / "fold_metrics.csv")
    if metrics is None:
        st.info(f"No {label} report yet for this run - run it from the sidebar.")
        return

    row = metrics.iloc[0]
    cols = st.columns(6)
    cols[0].metric("Test accuracy", f"{row['test_accuracy']:.3f}" if pd.notna(row["test_accuracy"]) else "n/a")
    cols[1].metric("Expectancy / trade", f"{row['test_expectancy']:.4f}" if pd.notna(row["test_expectancy"]) else "n/a")
    cols[2].metric("Final equity", f"${row['final_equity']:,.0f}")
    cols[3].metric("Max drawdown", f"{row['max_drawdown']:.2%}")
    sharpe = row["sharpe_ratio"] if "sharpe_ratio" in row and pd.notna(row["sharpe_ratio"]) else None
    cols[4].metric("Sharpe ratio", f"{sharpe:.2f}" if sharpe is not None else "n/a",
                   help="Annualized Sharpe ratio of the equity curve's daily returns (mean / std, "
                        "scaled by sqrt(252)), risk-free rate = 0. Higher is better; above ~1 is "
                        "considered decent for a systematic strategy. 'n/a' means the test window "
                        "was too short or had too few return observations to compute it.")
    cols[5].metric("Benchmark return", f"{row['benchmark_return']:.2%}")
    st.caption(f"n_test_trades={row['n_test_trades']}  |  n_admitted_portfolio={row['n_admitted_portfolio']}  |  "
               f"selected_threshold={row['selected_threshold']}")

    img_cols = st.columns(4)
    for col, name, caption in zip(
        img_cols,
        ["equity_curve.png", "drawdown.png", "calibration.png", "outcome_variance.png"],
        ["Equity curve", "Drawdown", "Calibration", "Outcome variance"],
    ):
        p = reports_dir / name
        if p.exists():
            col.image(str(p), caption=caption, use_container_width=True)

    variance_by_bucket = load_csv(reports_dir / "variance_by_bucket.csv")
    if variance_by_bucket is not None and len(variance_by_bucket):
        std_col = variance_by_bucket["std_return"].dropna()
        rising = len(std_col) >= 2 and std_col.iloc[-1] > std_col.iloc[0] * 1.2
        falling = len(std_col) >= 2 and std_col.iloc[-1] < std_col.iloc[0] * 0.8
        if rising:
            variance_verdict = ("⚠️ Outcome variance is **higher in the high-confidence buckets** than "
                                "the low-confidence ones - the model's most-trusted signals are actually "
                                "its least predictable in terms of win/loss size. Confidence here is "
                                "tracking direction, not risk.")
        elif falling:
            variance_verdict = ("✅ Outcome variance **shrinks as take_prob rises** - high-confidence "
                                "signals aren't just more often right, their outcomes are also more "
                                "consistent/tighter, which is what you'd want from a well-behaved "
                                "probability.")
        else:
            variance_verdict = ("ℹ️ Outcome variance is roughly flat across confidence buckets - "
                                "take_prob is tracking win-rate (see the calibration chart) more than "
                                "it's tracking the size/consistency of the outcome.")
        st.markdown(
            "**Outcome variance chart, explained:** the calibration chart shows whether a bucket's "
            "trades *win more often* as take_prob rises. This chart asks a different question - when "
            "a trade in that bucket resolves, how much does the realized `net_return` **vary**? A "
            "bucket with a tall bar has trades whose outcomes range widely (some big winners, some "
            "big losers cancelling out in the mean); a short bar means outcomes in that bucket are "
            "clustered close together. Reading left (low confidence) to right (high confidence) tells "
            "you whether the model's confidence says anything about the *risk* of a trade, not just "
            "its *direction*.\n\n" + variance_verdict
        )
        with st.expander("Variance by bucket (raw numbers)"):
            st.dataframe(variance_by_bucket, use_container_width=True)

    trades = load_csv(reports_dir / "accepted_trades.csv")
    if trades is not None and len(trades):
        st.markdown("**Best trades** (sorted by net_return)")
        st.dataframe(trades.head(15), use_container_width=True)
        st.markdown("**Worst trades**")
        st.dataframe(trades.tail(15).iloc[::-1], use_container_width=True)
    else:
        st.caption("No accepted trades above the selected threshold in this period.")


def alpaca_latest_trades(symbols: list[str]) -> dict:
    from real_data import _load_alpaca_env  # same loader cli.py's data pull uses
    key, secret = _load_alpaca_env()
    url = f"https://data.alpaca.markets/v2/stocks/trades/latest?symbols={','.join(symbols)}"
    req = urllib.request.Request(url, headers={
        "APCA-API-KEY-ID": key,
        "APCA-API-SECRET-KEY": secret,
    })
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Alpaca data API error {e.code}: {e.read().decode()[:300]}") from e


# ---------------------------------------------------------------------------
# sidebar - the full parameter form
# ---------------------------------------------------------------------------

st.sidebar.header("Run setup")
run_id = st.sidebar.text_input(
    "Run ID", value="intc_nvda_full",
    help="Folder name under runs/<run_id>/ where every artifact (model.pt, "
         "reports, plots) is saved. Reusing an existing run_id overwrites "
         "that run's prior artifacts.",
)
device = st.sidebar.selectbox(
    "Device", ["cuda", "cpu"], index=0,
    help="Where the GRU trains/runs. 'cuda' needs an NVIDIA GPU with a "
         "CUDA-enabled PyTorch install; cli.py falls back to CPU "
         "automatically if cuda was requested but isn't actually available.",
)

st.sidebar.subheader("Universe")
tickers_raw = st.sidebar.text_input(
    "Tickers (comma-separated)", value="INTC, NVDA",
    help="Stocks to backtest, e.g. 'AAPL, NVDA'. Each ticker gets its own "
         "regime/volatility fit and feature frame, but all tickers' trades "
         "are pooled into one shared GRU and one shared portfolio.",
)
tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
benchmark = st.sidebar.text_input(
    "Benchmark", value="SPY",
    help="Index/ETF used for the benchmark-trend and relative-strength "
         "entry rules (rules 1 & 3 in pullback_entry.py) and for the "
         "benchmark_return comparison shown in every report.",
)

st.sidebar.subheader("Model mode (feature blocks)")
mode = st.sidebar.selectbox(
    "Mode", list(MODE_TO_BLOCKS.keys()), index=0,
    help="Which feature blocks feed the GRU. Technical indicators are "
         "always on; regime (HMM bull/bear/sideways probabilities) and "
         "volatility (GARCH forecast) are togglable — more blocks means "
         "more columns the model has to learn from the same trade count, "
         "not automatically a better model.",
)

st.sidebar.subheader("Dates")
train_start = st.sidebar.date_input(
    "Train start", value=date(2015, 1, 1),
    help="Start of the GRU's training window — only trades with "
         "signal_date >= this are used to fit the model. Longer = more "
         "training trades, but also older, possibly less relevant regimes.",
)
train_end = st.sidebar.date_input(
    "Train end", value=date(2023, 6, 30),
    help="End of the training window. Must be strictly before 'Eval (dev) "
         "start' — the app blocks running training otherwise, since the "
         "splits can't overlap.",
)
dev_start = st.sidebar.date_input(
    "Eval (dev) start", value=date(2023, 7, 1),
    help="Start of the dev slice — held out from training, used only to "
         "pick the take_prob threshold and for early stopping. Also reused "
         "internally as this run's own 'test' slice for the Eval tab below "
         "(see cli.cmd_train) — it is NOT a genuine out-of-sample test; "
         "use the Test (holdout) dates further down for that.",
)
dev_end = st.sidebar.date_input(
    "Eval (dev) end", value=date(2025, 1, 1),
    help="End of the dev slice. Data is pulled only up to this date for "
         "training (see 'Data pulled from... to...' line above), so the "
         "Test (holdout) tab always does its own fresh data pull later.",
)
test_start = st.sidebar.date_input(
    "Test (holdout) start", value=date(2025, 1, 1),
    help="Start of a period the saved model never trained OR tuned its "
         "threshold on at all — used only by the 'Test (holdout)' tab's "
         "evaluate-holdout command. This is the only genuinely "
         "out-of-sample report in this app.",
)
test_end = st.sidebar.date_input(
    "Test (holdout) end", value=date.today(),
    help="End of the holdout window — defaults to today, i.e. 'everything "
         "since Test (holdout) start, up to now'.",
)
buffer_days = st.sidebar.number_input(
    "Warm-up buffer (days, before train/test start)", min_value=60, max_value=800, value=300, step=20,
    help="Extra history pulled before the window start so moving averages / the lookback window are valid on day 1.",
)
buffer_start = train_start - timedelta(days=buffer_days)
test_buffer_start = test_start - timedelta(days=buffer_days)

with st.sidebar.expander("Model hyperparameters"):
    lookback = st.number_input(
        "lookback", 5, 120, 30,
        help="How many prior trading sessions of feature history feed one "
             "GRU prediction. Candidates with fewer than this many prior "
             "rows are silently dropped — too large on a short-history "
             "ticker can shrink your usable trade count.",
    )
    hidden_size = st.number_input(
        "hidden_size", 4, 256, 32,
        help="GRU's hidden-state width — the model's capacity. Larger can "
             "fit more complex patterns but overfits faster given how few "
             "trades this pipeline typically produces (tens to low hundreds).",
    )
    dropout = st.slider(
        "dropout", 0.0, 0.8, 0.1,
        help="Fraction of GRU output zeroed during training. Higher fights "
             "overfitting on small datasets at the cost of slower, noisier "
             "convergence.",
    )
    lr = st.number_input(
        "lr", 1e-5, 1.0, 1e-3, format="%.5f",
        help="AdamW learning rate. Too high: loss oscillates or diverges "
             "(watch for a 'not converging' message). Too low: needs more "
             "epochs to get anywhere before max_epochs/patience cuts it off.",
    )
    weight_decay = st.number_input(
        "weight_decay", 0.0, 1.0, 1e-4, format="%.5f",
        help="AdamW's L2 penalty on the weights — another overfitting "
             "control; higher shrinks weights toward zero more aggressively.",
    )
    grad_clip = st.number_input(
        "grad_clip", 0.0, 10.0, 1.0,
        help="Max gradient norm before clipping — guards against exploding "
             "gradients destabilizing training. Rarely needs tuning unless "
             "loss spikes to huge values.",
    )
    max_epochs = st.number_input(
        "max_epochs", 1, 500, 50,
        help="Hard cap on training passes over the data. Training usually "
             "stops earlier via patience (below) unless dev loss keeps "
             "improving all the way to this cap.",
    )
    patience = st.number_input(
        "patience", 1, 50, 5,
        help="Early-stopping patience — epochs allowed with no val_loss "
             "improvement before training stops and the best-val_loss "
             "checkpoint is restored. Lower = faster/more conservative; "
             "higher = trains longer chasing small improvements.",
    )
    seed = st.number_input(
        "seed", 0, 10_000, 42,
        help="Random seed for model init + training shuffle order. Same "
             "seed + same data = reproducible results — change it to check "
             "whether a config change actually helped, or you just got a "
             "luckier draw.",
    )

with st.sidebar.expander("Pullback-entry rule"):
    st.caption("Defaults grid-searched for max candidate density on DAILY bars "
               "(not intraday — see README's day-trading limitation note). All 5 "
               "candidate_signal rules must fire on the SAME day, and each rule's "
               "pass rate barely moves with window size — shrinking these "
               "naively can give you FEWER candidates, not more.")
    ma_window = st.number_input(
        "ma_window", 3, 250, 30,
        help="Trend-filter moving-average length (rules 1 & 2: stock AND "
             "benchmark must both be above their own MA). Shorter reacts "
             "faster to trend changes but whipsaws more in choppy markets.",
    )
    rel_strength_window = st.number_input(
        "rel_strength_window", 2, 120, 10,
        help="Lookback in sessions for comparing the stock's trailing "
             "return to the benchmark's (rule 3) — the stock must be "
             "outperforming the benchmark over this window.",
    )
    pullback_window = st.number_input(
        "pullback_window", 1, 20, 3,
        help="How many of the most recent sessions (t-1..t-N) are checked "
             "for a dip below the prior high (rule 4) — the 'pullback' "
             "part of the setup.",
    )
    prior_high_lookback = st.number_input(
        "prior_high_lookback", 1, 60, 3,
        help="How far back close[t-4] must be the local max to count as a "
             "genuine 'prior high' (rule 4). Larger demands a more "
             "significant/older high; smaller accepts minor recent highs.",
    )
    min_rules_required = st.number_input(
        "min_rules_required (of 5)", 1, 5, 4,
        help="5 = strict AND of all 5 rules (original). Lower admits more "
             "candidates by requiring fewer rules to agree on the same day.",
    )

with st.sidebar.expander("Trade simulation"):
    atr_window = st.number_input(
        "atr_window", 2, 60, 7,
        help="Sessions averaged for the Average True Range (ATR) used to "
             "size the stop distance. Shorter reacts faster to recent "
             "volatility; longer smooths it out.",
    )
    stop_atr_mult = st.number_input(
        "stop_atr_mult", 0.1, 10.0, 2.5,
        help="Portfolio sizing requires stop_distance >= risk_pct_per_trade / "
             "max_position_pct as a % of price, or trades get rejected for "
             "oversized notional on expensive, low-ATR% tickers (e.g. AAPL). "
             "2.5 covers AAPL/NVDA/INTC at the defaults below.",
    )
    target_r_mult = st.number_input(
        "target_r_mult", 0.1, 10.0, 1.5,
        help="Profit target as a multiple of the ATR-based stop distance — "
             "e.g. 1.5 means the target is 1.5x further from entry than the "
             "stop. Higher asks for a bigger move before exiting "
             "profitably, which can lower the hit-rate within "
             "max_holding_sessions.",
    )
    max_holding_sessions = st.number_input(
        "max_holding_sessions", 1, 60, 3,
        help="Trade force-closes at this session's close ('timeout' exit) "
             "if neither stop nor target was hit first. Shorter = more "
             "day-trading-like, but gives price less time to reach target.",
    )
    commission_bps = st.number_input(
        "commission_bps", 0.0, 50.0, 1.0,
        help="Flat commission applied to every fill, in basis points of "
             "the fill price (1 bps = 0.01%). Set to your broker's actual "
             "rate for a realistic net_return.",
    )
    slippage_bps = st.number_input(
        "slippage_bps", 0.0, 50.0, 2.0,
        help="Assumed slippage per fill — entry pays it upward, exit pays "
             "it downward, in basis points. A simple fixed-cost model, not "
             "a market-impact model.",
    )

with st.sidebar.expander("Threshold search / portfolio"):
    min_trades_for_threshold = st.number_input(
        "min_trades_for_threshold", 1, 500, 10,
        help="A take_prob threshold is only accepted if it admits at "
             "least this many dev trades — guards against picking a "
             "threshold validated on a handful of lucky trades. Set too "
             "high relative to your actual trade count and training "
             "reports 'insufficient data'.",
    )
    threshold_lo = st.number_input(
        "threshold grid low", 0.0, 1.0, 0.2,
        help="Low end of the take_prob cutoff grid scanned on dev "
             "(np.linspace(low, high, steps)) to pick the threshold.",
    )
    threshold_hi = st.number_input(
        "threshold grid high", 0.0, 1.0, 0.9,
        help="High end of the same grid.",
    )
    threshold_n = st.number_input(
        "threshold grid steps", 2, 200, 25,
        help="How many points in the grid. The threshold with the best "
             "dev-set expectancy, among those clearing "
             "min_trades_for_threshold, wins.",
    )
    starting_cash = st.number_input(
        "starting_cash", 1_000.0, 10_000_000.0, 100_000.0,
        help="Portfolio's starting capital for the backtest — a scaling "
             "constant for dollar P&L figures; doesn't affect returns/ratios.",
    )
    risk_pct_per_trade = st.number_input(
        "risk_pct_per_trade", 0.001, 1.0, 0.01,
        help="Fraction of CURRENT equity risked per trade — position size "
             "= this x equity / stop_distance. Higher compounds gains/"
             "losses faster and sizes positions larger, which runs into "
             "max_position_pct more often.",
    )
    max_position_pct = st.number_input(
        "max_position_pct", 0.01, 1.0, 0.20,
        help="Single-position cap as a fraction of equity. A trade's "
             "notional can't exceed this — portfolio_backtest.py resizes "
             "shares down to fit it instead of skipping the trade. Too low "
             "on expensive, low-volatility tickers (small ATR as % of "
             "price) can shrink most positions well below the risk-sized "
             "target; see stop_atr_mult's note above.",
    )
    max_concurrent_trades = st.number_input(
        "max_concurrent_trades", 1, 50, 5,
        help="Max number of simultaneously open positions. Once this many "
             "are open, new candidates are skipped entirely — not resized "
             "— until a slot frees up.",
    )
    max_aggregate_risk_pct = st.number_input(
        "max_aggregate_risk_pct", 0.001, 1.0, 0.06,
        help="Cap on TOTAL open risk (sum of every open position's risk "
             "dollars) as a fraction of equity — limits how much can be at "
             "risk across all open trades at once, independent of the "
             "single-position cap above.",
    )
    n_states = st.number_input(
        "regime n_states", 2, 6, 3,
        help="Number of hidden states the regime HMM (causal_hmm.py) "
             "fits — 3 is the canonical bull/sideways/bear split. Only "
             "matters when 'regime' is in the selected Mode's feature "
             "blocks above.",
    )

form = dict(
    tickers=tickers, benchmark=benchmark, mode=mode,
    buffer_start=buffer_start, dev_end=dev_end,
    lookback=int(lookback), hidden_size=int(hidden_size), dropout=float(dropout),
    lr=float(lr), weight_decay=float(weight_decay), grad_clip=float(grad_clip),
    max_epochs=int(max_epochs), patience=int(patience), seed=int(seed),
    ma_window=int(ma_window), rel_strength_window=int(rel_strength_window),
    pullback_window=int(pullback_window), prior_high_lookback=int(prior_high_lookback),
    min_rules_required=int(min_rules_required),
    atr_window=int(atr_window), stop_atr_mult=float(stop_atr_mult), target_r_mult=float(target_r_mult),
    max_holding_sessions=int(max_holding_sessions), commission_bps=float(commission_bps),
    slippage_bps=float(slippage_bps),
    min_trades_for_threshold=int(min_trades_for_threshold),
    threshold_lo=float(threshold_lo), threshold_hi=float(threshold_hi), threshold_n=int(threshold_n),
    starting_cash=float(starting_cash), risk_pct_per_trade=float(risk_pct_per_trade),
    max_position_pct=float(max_position_pct), max_concurrent_trades=int(max_concurrent_trades),
    max_aggregate_risk_pct=float(max_aggregate_risk_pct), n_states=int(n_states),
)

run_dir = RUNS_DIR / run_id

# ---------------------------------------------------------------------------
# main area
# ---------------------------------------------------------------------------

st.title("Strategy Lab")
st.caption("Train/eval the GRU take-skip model on real tickers, then test the saved "
           "checkpoint on a later holdout period - all from one panel.")

tab_train, tab_eval, tab_test, tab_live, tab_predict = st.tabs(
    ["1. Train", "2. Eval (dev)", "3. Test (holdout)", "Live quote (Alpaca)", "Next-day prediction"]
)

with tab_train:
    st.write(f"Train window: **{train_start} -> {train_end}**  |  "
             f"Dev window: **{dev_start} -> {dev_end}**  |  "
             f"Data pulled from **{buffer_start}** (buffer) to **{dev_end}**.")
    st.write(f"Mode: **{mode}** -> feature_blocks = {MODE_TO_BLOCKS[mode]}")
    if st.button("Run training", type="primary"):
        if train_end >= dev_start:
            st.error("Train end must be before dev start - ranges can't overlap.")
        else:
            cfg = build_config(form)
            cfg_path = CONFIGS_DIR / f"{run_id}.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg))
            with st.spinner("Training on GPU..." if device == "cuda" else "Training on CPU..."):
                code, _ = run_cli([
                    "train", "--config", str(cfg_path), "--run-id", run_id, "--device", device,
                    "--train-start", train_start.isoformat(), "--train-end", train_end.isoformat(),
                    "--dev-start", dev_start.isoformat(), "--dev-end", dev_end.isoformat(),
                ])
            if code == 0:
                st.success("Training finished.")
            else:
                st.error(f"Training failed (exit code {code}) - see log above.")

    metadata = load_json(run_dir / "run_metadata.json")
    if metadata:
        st.subheader("Last training run")
        st.json({k: v for k, v in metadata.items() if k != "fold"}, expanded=False)

    training_curves_path = run_dir / "training_curves.png"
    if training_curves_path.exists():
        st.subheader("Training curves")
        st.image(str(training_curves_path), use_container_width=True)
        history = load_json(run_dir / "history.json")
        n_epochs = len(history["train_loss"]) if history else 0
        final_train_loss = history["train_loss"][-1] if history else None
        final_val_loss = history["val_loss"][-1] if history else None
        final_train_acc = history["train_acc"][-1] if history else None
        final_val_acc = history["val_acc"][-1] if history else None
        st.markdown(
            f"""
**How to read this:** the GRU sees every `(signal_date, label)` candidate once
per epoch and adjusts its weights to push its predicted take-probability
toward the actual outcome (label 1 = target hit profitably, 0 = stop/timeout).

- **Top panel (loss, lower is better):** `train_loss` is the model's error on
  the data it's currently learning from; `val_loss` is its error on the dev
  slice it never trains on — that gap is the real signal of whether the model
  is learning something that generalizes, or just memorizing the train set.
- **Bottom panel (accuracy, higher is better):** fraction of candidates where
  `take_prob >= 0.5` matched the actual label. The grey dashed line at 0.5 is
  chance level — a model that never learned anything sits on that line.
- Training stopped after **{n_epochs} epoch(s)** here, either because it hit
  `max_epochs` or because `val_loss` didn't improve for `patience` epochs in a
  row (early stopping) — whichever came first. The saved model (`model.pt`)
  is the checkpoint from the epoch with the **best val_loss seen**, not
  necessarily the last epoch run.
- Final epoch: `train_loss={final_train_loss:.4f}` `val_loss={final_val_loss:.4f}` `train_acc={final_train_acc:.3f}` `val_acc={final_val_acc:.3f}`
            """
            if history else ""
        )
    loss_variance_path = run_dir / "loss_variance.png"
    if loss_variance_path.exists():
        st.subheader("Per-trade loss variance per epoch")
        st.image(str(loss_variance_path), use_container_width=True)
        history = load_json(run_dir / "history.json")
        if history and "train_loss_std" in history and "val_loss_std" in history:
            final_train_loss_std = history["train_loss_std"][-1]
            final_val_loss_std = history["val_loss_std"][-1]
            rising = final_val_loss_std > history["val_loss_std"][0] * 1.2
            hetero_note = (
                "⚠️ Validation loss variance is **rising** as training goes on - the model is getting "
                "confidently wrong on a subset of trades (often the rarer/noisier setups) even while "
                "its *average* loss looks fine. Worth checking the worst trades in the eval report, or "
                "adding regularization."
                if rising else
                "✅ Loss variance is flat or shrinking over epochs - the model's error size is roughly "
                "**homoscedastic** (consistently sized across trades), not concentrated into a handful "
                "of badly-misjudged ones."
            )
            st.markdown(
                "**What this curve is:** the training-curves panel above shows the *average* loss per "
                "epoch, but two epochs with the same average loss can look very different underneath - "
                "one where every trade is off by about the same amount (**homoscedastic**, consistent "
                "error), or one where most trades are nearly perfect and a few are wildly wrong "
                "(**heteroscedastic**, lumpy error). This chart plots the **standard deviation of the "
                "per-trade loss** within each epoch to tell those two cases apart - distinct from the "
                "'Volatility clustering check' below, which is about GARCH fitting price volatility, "
                "not about the model's own error.\n\n"
                f"Final epoch: `train_loss_std={final_train_loss_std:.4f}` "
                f"`val_loss_std={final_val_loss_std:.4f}`.\n\n" + hetero_note
            )

    hetero_plots = sorted(run_dir.glob("heteroscedasticity_*.png")) if run_dir.exists() else []
    if hetero_plots:
        st.subheader("Volatility clustering check (train/dev)")
        for p in hetero_plots:
            st.image(str(p), caption=p.stem.replace("heteroscedasticity_", ""), use_container_width=True)

with tab_eval:
    st.write(f"Evaluates the saved model on the dev slice of its own training run "
             f"({dev_start} -> {dev_end}).")
    if st.button("Run evaluate (dev)"):
        if not run_dir.exists():
            st.error("No run found - train first.")
        else:
            with st.spinner("Evaluating..."):
                code, _ = run_cli(["evaluate", "--run-id", run_id, "--device", device])
            if code == 0:
                st.success("Evaluate finished.")
            else:
                st.error(f"Evaluate failed (exit code {code}) - see log above.")
    render_report(run_dir, "reports", "eval(dev)")

with tab_test:
    st.write(f"Scores the saved checkpoint on a period it never trained/tuned on: "
             f"**{test_start} -> {test_end}** (data pulled from {test_buffer_start} as buffer).")
    if st.button("Run holdout test", type="primary"):
        if not run_dir.exists():
            st.error("No run found - train first.")
        else:
            with st.spinner("Scoring holdout period..."):
                code, _ = run_cli([
                    "evaluate-holdout", "--run-id", run_id, "--device", device,
                    "--buffer-start", test_buffer_start.isoformat(),
                    "--test-start", test_start.isoformat(), "--test-end", test_end.isoformat(),
                ])
            if code == 0:
                st.success("Holdout test finished.")
            else:
                st.error(f"Holdout test failed (exit code {code}) - see log above.")
    render_report(run_dir, f"reports_holdout_{test_start.isoformat()}_{test_end.isoformat()}", "holdout test")

with tab_live:
    st.write("Pulls the latest trade price for each ticker from Alpaca's market-data API "
             "(same key/secret as the paper-trading adapter in AIStrategy/live_signal).")
    live_symbols = st.text_input(
        "Symbols", value=", ".join(tickers),
        help="Comma-separated tickers to fetch the latest trade price for, "
             "via Alpaca's market-data API — independent of the Universe "
             "tickers above; defaults to the same list but can be edited.",
    )
    if st.button("Fetch live quote"):
        syms = [s.strip().upper() for s in live_symbols.split(",") if s.strip()]
        try:
            data = alpaca_latest_trades(syms)
            rows = []
            for sym, trade in data.get("trades", {}).items():
                rows.append({"symbol": sym, "price": trade.get("p"), "size": trade.get("s"),
                             "timestamp": trade.get("t")})
            if rows:
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
            else:
                st.warning(f"No trade data returned: {data}")
        except RuntimeError as e:
            st.error(str(e))

with tab_predict:
    st.write(
        "Runs the pullback-entry rules + the saved GRU model **as of Test (holdout) end** "
        f"(**{test_end}**) and reports what that implies for the very next trading session "
        f"(**{(pd.Timestamp(test_end) + pd.tseries.offsets.BDay(1)).date()}**)."
    )
    st.caption(
        "This is not a crystal ball on tomorrow's price - every rule and every feature used here is "
        "computed from data up to and including the 'as of' date only. What it predicts is whether "
        "the *strategy itself* would buy at tomorrow's open, which is exactly how trade_simulation.py "
        "always enters: one session after the signal date."
    )
    as_of = st.date_input(
        "As of (signal date)", value=test_end,
        help="The last known trading day to evaluate the entry rules and model on. Defaults to Test "
             "(holdout) end. The prediction is always for the next trading session AFTER this date.",
    )
    predict_buffer_start = as_of - timedelta(days=buffer_days)
    if st.button("Predict next session", type="primary"):
        if not run_dir.exists():
            st.error("No run found - train first.")
        else:
            with st.spinner("Scoring the entry rules and model as of the selected date..."):
                code, _ = run_cli([
                    "predict-next", "--run-id", run_id, "--device", device,
                    "--buffer-start", predict_buffer_start.isoformat(),
                    "--as-of", as_of.isoformat(),
                ])
            if code == 0:
                st.success("Prediction finished.")
            else:
                st.error(f"Prediction failed (exit code {code}) - see log above.")

    pred_path = run_dir / "predictions" / f"{as_of.isoformat()}.json"
    predictions = load_json(pred_path) if pred_path.exists() else None
    if predictions is None:
        st.info("No prediction yet for this date - click 'Predict next session' above.")
    else:
        for p in predictions:
            st.markdown(f"### {p['ticker']}")
            if p["status"] != "ok":
                st.warning(p["note"])
                continue

            cols = st.columns(3)
            prob = p["take_prob"]
            cols[0].metric(
                "Model confidence (max likelihood of a good entry)",
                f"{prob:.1%}" if prob is not None else "n/a",
            )
            cols[1].metric("Selected threshold", f"{p['threshold']:.1%}")
            cols[2].metric(
                "Entry signal present?",
                f"{'Yes' if p['candidate_signal'] else 'No'} ({p['rules_passed']}/5, need "
                f">= {p['min_rules_required']})",
            )

            if p["decision"] == "BUY":
                st.success(f"**BUY — this is an entry (long) signal for {p['next_session']}.**")
            else:
                st.error(f"**NO BUY — not an entry signal for {p['next_session']}.**")
            st.write(p["reason"])

            plan = p.get("trade_plan")
            if plan is not None:
                st.markdown("**Trade plan**")
                plan_cols = st.columns(3)
                plan_cols[0].metric(
                    "Est. entry price",
                    f"${plan['est_entry_price']:,.2f}",
                    help=plan["est_entry_note"],
                )
                plan_cols[1].metric(
                    "Stop loss",
                    f"${plan['stop_loss_price']:,.2f}",
                    delta=f"{plan['stop_loss_pct']:.2%}",
                    delta_color="inverse",
                    help="Price (and % below estimated entry) at which the trade exits for a loss.",
                )
                plan_cols[2].metric(
                    "Take profit",
                    f"${plan['take_profit_price']:,.2f}",
                    delta=f"{plan['take_profit_pct']:.2%}",
                    help="Price (and % above estimated entry) at which the trade exits for a gain.",
                )
                st.caption(
                    f"Prices are estimates based on {plan['est_entry_note']}; the actual fill at "
                    f"{p['next_session']}'s open will differ slightly, but the stop/target *percentages* "
                    "are fixed by the trade-sim config (ATR-based stop, R-multiple target) and don't "
                    "depend on the exact entry price."
                )

            with st.expander("Rule-by-rule breakdown"):
                for rule, passed in p["rules"].items():
                    icon = "✅" if passed else "❌"
                    st.write(f"{icon} {RULE_LABELS.get(rule, rule)}")
