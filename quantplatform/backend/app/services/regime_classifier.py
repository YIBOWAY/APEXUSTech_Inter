"""Market regime classification based on VIX and XLK returns."""

import pandas as pd


def classify_vix_volatility(vix_level: float) -> str:
    """Classify volatility regime by VIX level (5-tier)."""
    if pd.isna(vix_level):
        return "Unknown"
    if vix_level < 15:
        return "Very Low Vol"
    elif vix_level < 20:
        return "Low Vol"
    elif vix_level < 25:
        return "Moderate Vol"
    elif vix_level < 30:
        return "High Vol"
    else:
        return "Extreme Vol"


def classify_trend_12m(ret_12m: float) -> str:
    """Classify market trend by 12-month rolling XLK return (5-tier)."""
    if pd.isna(ret_12m):
        return "Unknown"
    if ret_12m > 0.20:
        return "Strong Bull"
    elif ret_12m > 0.05:
        return "Bull"
    elif ret_12m > -0.05:
        return "Sideways"
    elif ret_12m > -0.20:
        return "Bear"
    else:
        return "Strong Bear"


def classify_momentum_3m(ret_3m: float) -> str:
    """Classify short-term momentum by 3-month rolling return (5-tier)."""
    if pd.isna(ret_3m):
        return "Unknown"
    if ret_3m > 0.10:
        return "Strong Up"
    elif ret_3m > 0.02:
        return "Up"
    elif ret_3m > -0.02:
        return "Flat"
    elif ret_3m > -0.10:
        return "Down"
    else:
        return "Strong Down"


def calculate_trend_strength(
    xlk_3m: float, xlk_6m: float, xlk_12m: float
) -> str:
    """Multi-timeframe trend consensus (4-tier)."""
    count = sum([
        xlk_3m > 0 if not pd.isna(xlk_3m) else False,
        xlk_6m > 0 if not pd.isna(xlk_6m) else False,
        xlk_12m > 0 if not pd.isna(xlk_12m) else False,
    ])
    if count == 3:
        return "Strong Uptrend"
    elif count == 2:
        return "Uptrend"
    elif count == 1:
        return "Mixed"
    else:
        return "Downtrend"


def classify_all_regimes(
    vix_monthly: pd.Series,
    xlk_monthly_returns: pd.Series,
) -> pd.DataFrame:
    """Classify all regime dimensions for a monthly time series.

    Args:
        vix_monthly: Monthly VIX values (month-end)
        xlk_monthly_returns: Monthly XLK returns

    Returns:
        DataFrame with columns: volatility_regime, market_trend,
        short_momentum, trend_strength, combined_regime
    """
    # Rolling returns (shifted to avoid lookahead bias)
    xlk_3m = xlk_monthly_returns.rolling(3).sum().shift(1)
    xlk_6m = xlk_monthly_returns.rolling(6).sum().shift(1)
    xlk_12m = xlk_monthly_returns.rolling(12).sum().shift(1)

    regimes = pd.DataFrame(index=xlk_monthly_returns.index)
    regimes["volatility_regime"] = vix_monthly.apply(classify_vix_volatility)
    regimes["market_trend"] = xlk_12m.apply(classify_trend_12m)
    regimes["short_momentum"] = xlk_3m.apply(classify_momentum_3m)
    regimes["trend_strength"] = pd.DataFrame({
        "xlk_3m": xlk_3m, "xlk_6m": xlk_6m, "xlk_12m": xlk_12m
    }).apply(lambda r: calculate_trend_strength(r["xlk_3m"], r["xlk_6m"], r["xlk_12m"]), axis=1)
    regimes["combined_regime"] = regimes["volatility_regime"] + " + " + regimes["market_trend"]

    return regimes
