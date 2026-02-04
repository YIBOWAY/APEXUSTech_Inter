"""
Integration helpers for using the optimized backtest in Project3 notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .optimized_backtest import OptimizedBacktest


@dataclass
class OptimizedBacktestResult:
    performance: dict
    spread: pd.Series
    z_scores: pd.Series
    backtest: OptimizedBacktest


def load_spread_series(csv_path: str) -> pd.Series:
    """Load spread series (FAR-NEAR) from a CSV with Date index."""
    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)
    if "NEAR" not in df.columns or "FAR" not in df.columns:
        raise ValueError(
            "CSV must contain 'NEAR' and 'FAR' columns. "
            f"Current columns: {df.columns.tolist()}"
        )
    spread = df["FAR"] - df["NEAR"]
    spread.name = "SPREAD"
    return spread


def run_optimized_backtest(
    csv_path: str,
    symbol: str,
    initial_capital: float,
    quantity: int,
    lookback_window: int,
    z_threshold: float,
    commission: float = 5.0,
    slippage: float = 0.01,
) -> OptimizedBacktestResult:
    """Run optimized backtest and return performance + series for plotting."""
    backtest = OptimizedBacktest(
        csv_path=csv_path,
        symbol=symbol,
        initial_capital=initial_capital,
        quantity=quantity,
        lookback_window=lookback_window,
        z_threshold=z_threshold,
        commission=commission,
        slippage=slippage,
    )
    performance = backtest.simulate_trading()

    spread = load_spread_series(csv_path)
    if backtest.strategy.z_scores:
        z_scores = pd.Series(backtest.strategy.z_scores).sort_index()
        z_scores.name = "Z_SCORE"
    else:
        z_scores = pd.Series(dtype=float, name="Z_SCORE")

    return OptimizedBacktestResult(
        performance=performance,
        spread=spread,
        z_scores=z_scores,
        backtest=backtest,
    )


def plot_optimized_performance(
    result: OptimizedBacktestResult,
    lookback_window: int,
    z_threshold: float,
    title: str,
):
    """Plot equity curve, spread/mean, and z-score for optimized run."""
    performance = result.performance
    curve = performance["equity_curve"]
    spread = result.spread

    fig = plt.figure(figsize=(12, 16))
    fig.suptitle(title, fontsize=16)

    ax1 = fig.add_subplot(311)
    ax1.plot(curve["equity_curve"], label="Equity Curve")
    ax1.set_title("Portfolio Equity Curve")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(True)
    ax1.legend()

    ax2 = fig.add_subplot(312)
    rolling_mean = spread.rolling(window=lookback_window).mean()
    ax2.plot(spread.index, spread.values, label="Spread (Far - Near)")
    ax2.plot(
        rolling_mean.index,
        rolling_mean.values,
        label=f"{lookback_window}-Day Rolling Mean",
        linestyle="--",
    )
    ax2.set_title("Spread and Rolling Mean")
    ax2.set_ylabel("Price Difference")
    ax2.grid(True)
    ax2.legend()

    ax3 = fig.add_subplot(313)
    if result.z_scores.empty:
        rolling_std = spread.rolling(window=lookback_window).std()
        z_scores = (spread - rolling_mean) / rolling_std
    else:
        z_scores = result.z_scores
    ax3.plot(z_scores.index, z_scores.values, label="Z-Score")
    ax3.axhline(z_threshold, color="r", linestyle="--", label=f"Threshold ({z_threshold})")
    ax3.axhline(-z_threshold, color="r", linestyle="--")
    ax3.axhline(0.0, color="k", linestyle="-")

    ax3.set_title("Spread Z-Score")
    ax3.set_ylabel("Z-Score")
    ax3.set_xlabel("Date")
    ax3.grid(True)
    ax3.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def summarize_performance(result: OptimizedBacktestResult):
    """Print key metrics from optimized performance dict."""
    performance = result.performance
    curve = performance["equity_curve"]
    total_return = performance["total_return"]
    sharpe_ratio = performance["sharpe_ratio"]
    max_drawdown = performance["max_drawdown"]

    print(f"Total Return: {total_return:.2%}")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")
    print(f"Final Value: ${curve['total'].iloc[-1]:,.0f}")
