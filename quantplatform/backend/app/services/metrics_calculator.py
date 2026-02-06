"""Performance metrics calculation and equity curve generation."""

import numpy as np
import pandas as pd


def calculate_metrics(monthly_returns: pd.Series) -> dict:
    """Calculate full performance metrics from monthly return series.

    Args:
        monthly_returns: pd.Series of monthly strategy returns (e.g. 0.023 = 2.3%)

    Returns:
        dict with keys: total_return, annual_return, annual_volatility,
                        sharpe_ratio, max_drawdown, win_rate, profit_factor, total_months
    """
    if monthly_returns.empty:
        return {
            "total_return": 0.0,
            "annual_return": 0.0,
            "annual_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "total_months": 0,
        }

    monthly_mean = monthly_returns.mean()
    monthly_std = monthly_returns.std()

    # Annualize
    annual_return = (1 + monthly_mean) ** 12 - 1
    annual_vol = monthly_std * np.sqrt(12)
    sharpe = annual_return / annual_vol if annual_vol > 0 else 0.0

    # Cumulative return
    cumulative = (1 + monthly_returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)

    # Max drawdown
    running_max = cumulative.cummax()
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = float(drawdowns.min())

    # Win rate
    win_rate = float((monthly_returns > 0).mean())

    # Profit factor
    gains = monthly_returns[monthly_returns > 0].sum()
    losses = abs(monthly_returns[monthly_returns < 0].sum())
    profit_factor = float(gains / losses) if losses > 0 else 999.0

    return {
        "total_return": round(total_return, 6),
        "annual_return": round(float(annual_return), 6),
        "annual_volatility": round(float(annual_vol), 6),
        "sharpe_ratio": round(float(sharpe), 4),
        "max_drawdown": round(max_drawdown, 6),
        "win_rate": round(win_rate, 4),
        "profit_factor": round(profit_factor, 4),
        "total_months": len(monthly_returns),
    }


def build_equity_curve(
    monthly_returns: pd.Series,
    benchmark_prices: pd.Series | None = None,
    start_value: float = 10000.0,
) -> list[dict]:
    """Convert monthly returns to EquityPoint list for frontend charts.

    Returns list of {"date": "2020-01-31", "value": 10230.0, "benchmark": 10150.0}
    """
    if monthly_returns.empty:
        return []

    # Strategy equity curve
    cumulative = (1 + monthly_returns).cumprod()
    strategy_values = cumulative * start_value

    # Benchmark curve - benchmark_prices is already aligned to strategy dates
    if benchmark_prices is not None and not benchmark_prices.empty:
        # Reindex to strategy dates, forward-fill any gaps
        bench_aligned = benchmark_prices.reindex(strategy_values.index, method="ffill")
        bench_aligned = bench_aligned.dropna()
        if len(bench_aligned) > 0:
            # Normalize: start at start_value
            bench_normalized = bench_aligned / bench_aligned.iloc[0] * start_value
        else:
            bench_normalized = pd.Series(start_value, index=strategy_values.index)
    else:
        bench_normalized = pd.Series(start_value, index=strategy_values.index)

    # Build output list
    points = []
    for dt in strategy_values.index:
        bench_val = float(bench_normalized.get(dt, start_value))
        points.append({
            "date": dt.strftime("%Y-%m-%d"),
            "value": round(float(strategy_values[dt]), 2),
            "benchmark": round(bench_val, 2),
        })

    return points
