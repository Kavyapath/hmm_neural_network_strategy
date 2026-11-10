"""Portfolio-level backtest. Wraps ALREADY-SIMULATED per-trade outcomes
(Step 5's simulate_trade results, with each trade's take_prob attached from
Step 6/7/8) with capital accounting: admission control, position sizing,
and an equity curve built from actual cash flows, not summed percentages.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    starting_cash: float = 100_000.0
    risk_pct_per_trade: float = 0.01       # fraction of CURRENT equity, recomputed per admission
    max_position_pct: float = 0.20          # single-position cap, fraction of equity
    max_concurrent_trades: int = 5
    max_aggregate_risk_pct: float = 0.06    # sum of open-position risk, fraction of equity


@dataclass
class OpenPosition:
    ticker: str
    shares: float
    entry_price: float
    entry_date: pd.Timestamp
    stop: float
    exit_date: pd.Timestamp
    exit_price: float
    risk_dollars: float


def rank_same_day_candidates(candidates: pd.DataFrame) -> pd.DataFrame:
    """candidates must have columns: entry_date, ticker, take_prob, ...
    Deterministic order: take_prob desc, ticker asc, within each entry_date."""
    return candidates.sort_values(
        by=["entry_date", "take_prob", "ticker"], ascending=[True, False, True]
    ).reset_index(drop=True)


def run_portfolio_backtest(
    accepted_candidates: pd.DataFrame,   # already filtered by threshold; has entry_date, exit_date,
                                           # ticker, entry_price (raw), stop, net_return (per Step 5), take_prob
    cfg: PortfolioConfig,
) -> dict:
    candidates = rank_same_day_candidates(accepted_candidates)
    cash = cfg.starting_cash
    equity = cfg.starting_cash
    open_positions: list[OpenPosition] = []
    closed_trades = []
    equity_curve = []   # (date, equity) checkpoints at every entry/exit event

    all_events = sorted(
        set(candidates["entry_date"]) | set(candidates["exit_date"].dropna()),
    )

    for event_date in all_events:
        # 1. Close any positions exiting today — release cash, realize P&L.
        still_open = []
        for pos in open_positions:
            if pos.exit_date == event_date:
                proceeds = pos.shares * pos.exit_price
                cash += proceeds
                equity = cash + sum(p.shares * p.entry_price for p in still_open)
                closed_trades.append(pos)
            else:
                still_open.append(pos)
        open_positions = still_open

        # 2. Admit new candidates signaling entry today, in deterministic order.
        todays_candidates = candidates[candidates["entry_date"] == event_date]
        for _, cand in todays_candidates.iterrows():
            if any(p.ticker == cand["ticker"] for p in open_positions):
                continue  # overlapping signal in same asset — skip, explicit rule

            stop_distance = cand["entry_price"] - cand["stop"]
            if stop_distance <= 0:
                continue  # malformed candidate, should not happen if Step 5 is correct — skip defensively
            if len(open_positions) >= cfg.max_concurrent_trades:
                continue  # a count cap — nothing to resize, there's no open slot

            open_risk = sum(p.risk_dollars for p in open_positions)
            risk_room = cfg.max_aggregate_risk_pct * equity - open_risk
            if risk_room <= 0:
                continue  # aggregate risk budget already fully committed

            # Size by risk first, then resize DOWN (never up) to also fit
            # inside cash on hand, the single-position notional cap, and the
            # remaining aggregate-risk room. A tight stop on an expensive
            # ticker (small stop_distance, high entry_price) used to make
            # `shares` blow past max_position_pct and get skipped outright —
            # now it gets admitted at a smaller size instead.
            risk_budget = cfg.risk_pct_per_trade * equity
            risk_dollars = min(risk_budget, risk_room)
            shares = risk_dollars / stop_distance
            notional = shares * cand["entry_price"]

            max_notional = min(cash, cfg.max_position_pct * equity)
            if notional > max_notional:
                shares = max_notional / cand["entry_price"]
                notional = shares * cand["entry_price"]
                risk_dollars = shares * stop_distance

            if shares <= 0:
                continue  # nothing left to size into (e.g. cash exhausted)

            cash -= notional
            open_positions.append(OpenPosition(
                ticker=cand["ticker"], shares=shares, entry_price=cand["entry_price"],
                entry_date=cand["entry_date"], stop=cand["stop"], exit_date=cand["exit_date"],
                exit_price=cand["exit_price"], risk_dollars=risk_dollars,
            ))

        equity = cash + sum(p.shares * p.entry_price for p in open_positions)
        equity_curve.append((event_date, equity))

    equity_series = pd.Series(
        [e for _, e in equity_curve], index=pd.DatetimeIndex([d for d, _ in equity_curve])
    )
    return {
        "equity_curve": equity_series,
        "closed_trades": closed_trades,
        "n_accepted": len(candidates),
        "n_admitted": len(closed_trades),
        "final_equity": equity_series.iloc[-1] if len(equity_series) else cfg.starting_cash,
    }

