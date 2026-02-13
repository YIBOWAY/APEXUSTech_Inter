"""Shared fixtures for strategy unit tests.

Provides pre-built DataFrames of synthetic price/volume data
so each test module can focus on logic, not data construction.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def date_index():
    """~2 years of business-day dates (2020-01 to 2021-12)."""
    return pd.bdate_range("2020-01-02", "2021-12-31", freq="B")


@pytest.fixture
def simple_adj_close(date_index):
    """15 synthetic stocks + XLK benchmark + ^IRX risk-free rate.

    calculate_momentum_scores() requires >= 10 valid stocks.
    Stocks 01-05 trend up, 06-10 flat, 11-15 trend down.
    """
    np.random.seed(42)
    n = len(date_index)

    data = {}
    # 5 uptrending stocks
    for i, drift in enumerate([0.0015, 0.0012, 0.001, 0.0008, 0.0006], start=1):
        data[f"STOCK_{i:02d}"] = 100 * np.cumprod(1 + np.random.normal(drift, 0.01, n))
    # 5 flat stocks
    for i in range(6, 11):
        data[f"STOCK_{i:02d}"] = 100 * np.cumprod(1 + np.random.normal(0.0, 0.01, n))
    # 5 downtrending stocks
    for i, drift in enumerate([-0.0005, -0.0008, -0.001, -0.0012, -0.0015], start=11):
        data[f"STOCK_{i:02d}"] = 100 * np.cumprod(1 + np.random.normal(drift, 0.01, n))

    data["XLK"] = 100 * np.cumprod(1 + np.random.normal(0.0004, 0.008, n))
    data["^VIX"] = np.random.uniform(12, 30, n)
    data["^IRX"] = np.random.uniform(4.0, 5.5, n)  # T-Bill rate ~4-5.5%

    return pd.DataFrame(data, index=date_index)


@pytest.fixture
def simple_volume(date_index):
    """Volume data matching simple_adj_close stock tickers."""
    np.random.seed(43)
    n = len(date_index)

    data = {}
    for i in range(1, 16):
        data[f"STOCK_{i:02d}"] = np.random.randint(1_000_000, 10_000_000, n)
    data["XLK"] = np.random.randint(5_000_000, 20_000_000, n)

    return pd.DataFrame(data, index=date_index)


@pytest.fixture
def known_monthly_returns():
    """Hand-crafted monthly returns with known outcomes.

    12 months: [+5%, -3%, +2%, +4%, -1%, +3%, -2%, +6%, +1%, -4%, +3%, +2%]
    Total compound return = prod(1+r) - 1 = 0.16068...
    """
    dates = pd.date_range("2020-01-31", periods=12, freq="ME")
    returns = [0.05, -0.03, 0.02, 0.04, -0.01, 0.03, -0.02, 0.06, 0.01, -0.04, 0.03, 0.02]
    return pd.Series(returns, index=dates, name="Strategy Returns")
