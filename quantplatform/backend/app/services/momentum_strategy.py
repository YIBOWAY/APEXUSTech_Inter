"""Core momentum strategy logic extracted from Project 4 V2 notebook."""

import numpy as np
import pandas as pd

from app.config import settings

# Use single source of truth from config
XLK_TICKERS = settings.XLK_TICKERS
EXCLUDE_SYMBOLS = settings.BENCHMARK_TICKERS


def calculate_momentum_scores(
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    daily_returns: pd.DataFrame,
    previous_date: pd.Timestamp,
    lookback_months: int,
    method: str = "simple",
) -> pd.Series:
    """
    Calculates momentum scores for all valid stocks on a given date.

    Parameters:
    - adj_close: DataFrame of adjusted close prices
    - volume: DataFrame of trading volumes
    - daily_returns: DataFrame of daily returns (adj_close.pct_change())
    - previous_date: The date for calculation (end of the previous month)
    - lookback_months: The lookback period in months (3, 6, 12)
    - method: 'simple', 'risk_adjusted', or 'volume_weighted'

    Returns:
    - A Series of scores indexed by ticker (higher is better momentum)
    """
    # Handle timezone issues
    if adj_close.index.tz is not None:
        adj_close_tz_naive = adj_close.copy()
        adj_close_tz_naive.index = adj_close.index.tz_localize(None)
    else:
        adj_close_tz_naive = adj_close

    if volume.index.tz is not None:
        volume_tz_naive = volume.copy()
        volume_tz_naive.index = volume.index.tz_localize(None)
    else:
        volume_tz_naive = volume

    if daily_returns.index.tz is not None:
        daily_returns_tz_naive = daily_returns.copy()
        daily_returns_tz_naive.index = daily_returns.index.tz_localize(None)
    else:
        daily_returns_tz_naive = daily_returns

    if hasattr(previous_date, "tz") and previous_date.tz is not None:
        previous_date = previous_date.tz_localize(None)

    # Find lookback start date
    lookback_start = previous_date - pd.DateOffset(months=lookback_months)
    available_dates = adj_close_tz_naive.index[adj_close_tz_naive.index >= lookback_start]
    if len(available_dates) == 0:
        return pd.Series(dtype=float)
    lookback_start = available_dates[0]

    # Extract lookback period data
    close_lookback = adj_close_tz_naive.loc[lookback_start:previous_date]

    # Filter valid stocks
    valid_stocks = []
    for col in close_lookback.columns:
        if col not in EXCLUDE_SYMBOLS and close_lookback[col].notna().all():
            valid_stocks.append(col)
    valid_stocks = pd.Index(valid_stocks)

    if len(valid_stocks) < 10:
        return pd.Series(dtype=float)

    # Basic momentum: price rate of change
    try:
        close_t = adj_close_tz_naive.loc[previous_date, valid_stocks]
        close_tk = adj_close_tz_naive.loc[lookback_start, valid_stocks]
        mom = close_t / close_tk - 1
    except KeyError:
        return pd.Series(dtype=float)

    if method == "simple":
        scores = mom

    elif method == "risk_adjusted":
        daily_ret_lb = daily_returns_tz_naive.loc[lookback_start:previous_date, valid_stocks]
        vol = daily_ret_lb.std() * np.sqrt(252)  # Annualize
        vol = vol.replace(0, np.nan)
        vol = vol.where(vol > 0.01, np.nan)
        scores = mom / vol
        scores = scores.where(np.abs(scores) <= 10, np.nan)

    elif method == "volume_weighted":
        baseline_start = previous_date - pd.DateOffset(months=12)
        available_baseline_dates = adj_close_tz_naive.index[
            adj_close_tz_naive.index >= baseline_start
        ]
        if len(available_baseline_dates) == 0:
            baseline_start = adj_close_tz_naive.index[0]
        else:
            baseline_start = available_baseline_dates[0]

        volume_valid_stocks = valid_stocks.intersection(volume_tz_naive.columns)
        vol_lb = volume_tz_naive.loc[lookback_start:previous_date, volume_valid_stocks].mean()
        vol_baseline = volume_tz_naive.loc[
            baseline_start:previous_date, volume_valid_stocks
        ].mean()

        vol_baseline = vol_baseline.replace(0, np.nan)
        volume_factor = vol_lb / vol_baseline
        volume_factor = volume_factor.where(
            (volume_factor >= 0.1) & (volume_factor <= 10), 1.0
        )

        scores = pd.Series(index=valid_stocks, dtype=float)
        for stock in volume_valid_stocks:
            if (
                stock in mom.index
                and not pd.isna(volume_factor[stock])
                and volume_factor[stock] > 0
            ):
                scores[stock] = mom[stock] * volume_factor[stock]
            elif stock in mom.index:
                scores[stock] = mom[stock]
        scores = scores.dropna()

    else:
        raise ValueError(f"Invalid method: {method}")

    # Cleanup
    scores = scores.replace([np.inf, -np.inf], np.nan)
    if len(scores.dropna()) > 5:
        mean_score = scores.mean()
        std_score = scores.std()
        if std_score > 0:
            scores = scores.where(np.abs(scores - mean_score) <= 3 * std_score, np.nan)

    return scores.dropna()


def backtest_strategy(
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    lookback_months: int = 6,
    method: str = "simple",
    decile: float = 0.2,
    tc_bps: int = 5,
    winsor_q: float = 0.01,
) -> tuple[pd.Series, list[dict]]:
    """
    Backtests a long-only momentum strategy with monthly rebalancing.

    Selects the top `decile` stocks by momentum score each month,
    equal-weighted, with transaction costs.

    Parameters:
    - adj_close: DataFrame of adjusted close prices
    - volume: DataFrame of trading volumes
    - lookback_months: Lookback period (3, 6, 12)
    - method: 'simple', 'risk_adjusted', or 'volume_weighted'
    - decile: The top quantile to go long (0.2 = top 20%)
    - tc_bps: Transaction cost per side in basis points
    - winsor_q: Winsorization quantile

    Returns:
    - Tuple of (monthly_returns Series, trade_signals list)
    """
    daily_returns = adj_close.pct_change()

    # Get month-end dates
    month_ends = adj_close.resample("ME").last().index

    # Filter for valid month-end dates
    valid_month_ends = []
    for me in month_ends:
        month_data = adj_close.loc[adj_close.index.month == me.month]
        month_data = month_data.loc[month_data.index.year == me.year]
        if len(month_data) > 0:
            actual_month_end = month_data.index[-1]
            valid_month_ends.append(actual_month_end)
    valid_month_ends = pd.DatetimeIndex(valid_month_ends).unique()

    # Initialize returns Series
    portfolio_returns = pd.Series(
        index=valid_month_ends[lookback_months:], dtype=float, name="Strategy Returns"
    )
    trade_signals: list[dict] = []
    prev_long_stocks: list[str] = []

    # Monthly rebalancing loop
    for i in range(lookback_months, len(valid_month_ends)):
        current_date = valid_month_ends[i]
        previous_date = valid_month_ends[i - 1]

        if current_date not in adj_close.index or previous_date not in adj_close.index:
            portfolio_returns[current_date] = 0.0
            continue

        # Calculate scores from previous month-end
        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date, lookback_months, method
        )

        if scores.empty:
            portfolio_returns[current_date] = 0.0
            continue

        # Rank and select top decile for long
        num_stocks = len(scores)
        top_n = max(1, int(num_stocks * decile))

        ranks = scores.rank(ascending=False)
        long_stocks = scores[ranks <= top_n].index.tolist()

        # Record trade signals (BUY for new positions)
        for ticker in long_stocks:
            price = adj_close.loc[previous_date, ticker] if ticker in adj_close.columns else None
            trade_signals.append({
                "signal_date": previous_date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "direction": "BUY",
                "weight": 1.0 / len(long_stocks),
                "score": float(scores[ticker]) if ticker in scores else None,
                "price": float(price) if price is not None else None,
            })

        # Next month's return
        try:
            ret_next = (
                adj_close.loc[current_date, scores.index]
                / adj_close.loc[previous_date, scores.index]
                - 1
            )
            ret_next = ret_next.replace([np.inf, -np.inf], np.nan).dropna()
        except KeyError:
            portfolio_returns[current_date] = 0.0
            continue

        # Winsorize
        if len(ret_next) > 5:
            lower = ret_next.quantile(winsor_q)
            upper = ret_next.quantile(1 - winsor_q)
            ret_next = ret_next.clip(lower, upper)

        # Calculate long-only returns
        available_long = [s for s in long_stocks if s in ret_next.index]

        if len(available_long) == 0:
            portfolio_returns[current_date] = 0.0
            continue

        long_ret = ret_next[available_long].mean()

        if pd.isna(long_ret):
            portfolio_returns[current_date] = 0.0
            continue

        # Transaction cost: estimate turnover from portfolio changes
        turnover = len(set(long_stocks) - set(prev_long_stocks)) / max(len(long_stocks), 1)
        tc = (tc_bps / 10000) * 2 * turnover  # Only pay TC on changed positions
        strategy_ret = long_ret - tc

        portfolio_returns[current_date] = strategy_ret
        prev_long_stocks = long_stocks

    return portfolio_returns.dropna(), trade_signals


def run_full_backtest(
    adj_close: pd.DataFrame,
    volume: pd.DataFrame,
    benchmark_ticker: str = "XLK",
    lookback_months: int = 6,
    method: str = "simple",
    decile: float = 0.2,
    tc_bps: int = 5,
    winsor_q: float = 0.01,
    start_value: float = 10000.0,
) -> dict:
    """
    Run full backtest and return all results in API-ready format.

    Returns dict with:
    - monthly_returns: list of {date, return}
    - equity_curve: list of {date, value, benchmark}
    - metrics: performance metrics dict
    - trade_signals: list of trade signal dicts
    """
    from app.services.metrics_calculator import build_equity_curve, calculate_metrics

    # Run backtest
    returns_series, trade_signals = backtest_strategy(
        adj_close=adj_close,
        volume=volume,
        lookback_months=lookback_months,
        method=method,
        decile=decile,
        tc_bps=tc_bps,
        winsor_q=winsor_q,
    )

    if returns_series.empty:
        return {
            "monthly_returns": [],
            "equity_curve": [],
            "metrics": {},
            "trade_signals": [],
        }

    # Extract annualized risk-free rate from ^IRX (13-week T-Bill, quoted in %)
    annual_rf = 0.0
    if "^IRX" in adj_close.columns:
        irx_series = adj_close["^IRX"].dropna()
        if not irx_series.empty:
            annual_rf = float(irx_series.mean()) / 100.0  # e.g. 4.5 -> 0.045

    # Calculate metrics
    metrics = calculate_metrics(returns_series, annual_risk_free_rate=annual_rf)

    # Build equity curve
    benchmark_prices = None
    if benchmark_ticker in adj_close.columns:
        benchmark_monthly = adj_close[benchmark_ticker].resample("ME").last()
        benchmark_monthly = benchmark_monthly.reindex(returns_series.index, method="ffill")
        benchmark_prices = benchmark_monthly

    equity_curve = build_equity_curve(returns_series, benchmark_prices, start_value)

    # Format monthly returns
    monthly_returns = [
        {"date": d.strftime("%Y-%m-%d"), "return": float(r)}
        for d, r in returns_series.items()
    ]

    return {
        "monthly_returns": monthly_returns,
        "equity_curve": equity_curve,
        "metrics": metrics,
        "trade_signals": trade_signals,
    }
