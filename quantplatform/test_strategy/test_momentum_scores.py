"""Unit tests for calculate_momentum_scores() in momentum_strategy.py.

Tests scoring logic for all three methods (simple, risk_adjusted, volume_weighted)
using synthetic data with predictable momentum rankings.
"""

import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.services.momentum_strategy import calculate_momentum_scores


@pytest.fixture
def score_data():
    """Build a dataset with 15 stocks where STOCK_01 has the best momentum.

    ~7 months of business days.
    Stocks 01-05: strong uptrend, Stocks 06-10: flat, Stocks 11-15: downtrend.
    calculate_momentum_scores() requires >= 10 valid stocks.
    """
    dates = pd.bdate_range("2020-01-02", periods=150, freq="B")
    np.random.seed(100)
    n = len(dates)

    data = {}
    # 5 uptrending stocks
    for i, drift in enumerate([0.0020, 0.0015, 0.0010, 0.0008, 0.0005], start=1):
        data[f"STOCK_{i:02d}"] = 100 * np.cumprod(1 + np.full(n, drift) + np.random.normal(0, 0.002, n))
    # 5 flat stocks
    for i in range(6, 11):
        data[f"STOCK_{i:02d}"] = 100 * np.cumprod(1 + np.random.normal(0, 0.002, n))
    # 5 downtrending stocks
    for i, drift in enumerate([-0.0005, -0.0008, -0.0010, -0.0015, -0.0020], start=11):
        data[f"STOCK_{i:02d}"] = 100 * np.cumprod(1 + np.full(n, drift) + np.random.normal(0, 0.002, n))

    adj_close = pd.DataFrame(data, index=dates)

    volume = pd.DataFrame(
        {k: np.random.randint(1_000_000, 5_000_000, n) for k in data},
        index=dates,
    )

    daily_returns = adj_close.pct_change()
    previous_date = dates[-1]

    return adj_close, volume, daily_returns, previous_date


class TestSimpleMethod:
    """Tests for method='simple' (price rate of change)."""

    def test_ranking_order(self, score_data):
        """Top scorer should be one of the uptrending stocks (01-05)."""
        adj_close, volume, daily_returns, previous_date = score_data
        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="simple"
        )
        assert not scores.empty
        top_stock = scores.idxmax()
        uptrending = {f"STOCK_{i:02d}" for i in range(1, 6)}
        assert top_stock in uptrending, f"Top stock {top_stock} not in uptrending group"

    def test_positive_scores_for_uptrending(self, score_data):
        """Stocks with positive drift should have positive scores."""
        adj_close, volume, daily_returns, previous_date = score_data
        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="simple"
        )
        assert not scores.empty
        assert scores["STOCK_01"] > 0
        assert scores["STOCK_02"] > 0

    def test_returns_series_type(self, score_data):
        """Scores should be a pd.Series of floats."""
        adj_close, volume, daily_returns, previous_date = score_data
        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="simple"
        )
        assert isinstance(scores, pd.Series)
        assert np.issubdtype(scores.dtype, np.floating)

    def test_score_count_matches_stocks(self, score_data):
        """Should return scores for all 15 stocks."""
        adj_close, volume, daily_returns, previous_date = score_data
        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="simple"
        )
        assert len(scores) == 15


class TestRiskAdjustedMethod:
    """Tests for method='risk_adjusted' (return / annualised vol)."""

    def test_returns_non_empty(self, score_data):
        """Should produce non-empty scores."""
        adj_close, volume, daily_returns, previous_date = score_data
        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="risk_adjusted"
        )
        assert not scores.empty

    def test_different_from_simple(self, score_data):
        """Risk-adjusted scores should differ in magnitude from simple scores."""
        adj_close, volume, daily_returns, previous_date = score_data
        simple = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="simple"
        )
        risk_adj = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="risk_adjusted"
        )
        assert not simple.empty and not risk_adj.empty
        common = simple.index.intersection(risk_adj.index)
        assert not np.allclose(simple[common].values, risk_adj[common].values)


class TestVolumeWeightedMethod:
    """Tests for method='volume_weighted' (return * volume ratio)."""

    def test_returns_scores(self, score_data):
        """Should return non-empty scores for volume_weighted."""
        adj_close, volume, daily_returns, previous_date = score_data
        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="volume_weighted"
        )
        assert not scores.empty
        assert len(scores) > 0


class TestEdgeCases:
    """Edge cases and error handling."""

    def test_insufficient_data(self):
        """Too few data points should return empty Series."""
        dates = pd.bdate_range("2020-01-02", periods=5, freq="B")
        adj_close = pd.DataFrame({"A": [100, 101, 102, 103, 104]}, index=dates)
        volume = pd.DataFrame({"A": [1e6] * 5}, index=dates)
        daily_returns = adj_close.pct_change()

        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, dates[-1],
            lookback_months=6, method="simple"
        )
        assert scores.empty

    def test_fewer_than_10_stocks_returns_empty(self):
        """Fewer than 10 valid stocks should return empty Series."""
        dates = pd.bdate_range("2020-01-02", periods=150, freq="B")
        np.random.seed(200)
        n = len(dates)

        adj_close = pd.DataFrame(
            {f"S{i}": 100 * np.cumprod(1 + np.random.normal(0, 0.01, n)) for i in range(5)},
            index=dates,
        )
        volume = pd.DataFrame(
            {f"S{i}": np.random.randint(1e6, 5e6, n) for i in range(5)},
            index=dates,
        )
        daily_returns = adj_close.pct_change()

        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, dates[-1],
            lookback_months=3, method="simple"
        )
        assert scores.empty

    def test_invalid_method_raises(self, score_data):
        """Invalid method name should raise ValueError."""
        adj_close, volume, daily_returns, previous_date = score_data
        with pytest.raises(ValueError, match="Invalid method"):
            calculate_momentum_scores(
                adj_close, volume, daily_returns, previous_date,
                lookback_months=3, method="invalid_method"
            )

    def test_excludes_benchmark_tickers(self, score_data):
        """XLK, ^VIX, ^IRX should never appear in scores."""
        adj_close, volume, daily_returns, previous_date = score_data
        adj_close["XLK"] = 100.0
        adj_close["^VIX"] = 20.0
        adj_close["^IRX"] = 4.5

        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="simple"
        )
        assert "XLK" not in scores.index
        assert "^VIX" not in scores.index
        assert "^IRX" not in scores.index

    def test_timezone_aware_index(self, score_data):
        """Should handle timezone-aware DatetimeIndex without error."""
        adj_close, volume, daily_returns, previous_date = score_data

        adj_close.index = adj_close.index.tz_localize("UTC")
        volume.index = volume.index.tz_localize("UTC")
        daily_returns.index = daily_returns.index.tz_localize("UTC")
        previous_date = adj_close.index[-1]

        scores = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="simple"
        )
        assert not scores.empty

    def test_different_lookback_periods(self, score_data):
        """Lookback 3 and 6 months should produce different score magnitudes."""
        adj_close, volume, daily_returns, previous_date = score_data
        scores_3m = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=3, method="simple"
        )
        scores_6m = calculate_momentum_scores(
            adj_close, volume, daily_returns, previous_date,
            lookback_months=6, method="simple"
        )
        if not scores_3m.empty and not scores_6m.empty:
            common = scores_3m.index.intersection(scores_6m.index)
            assert len(common) > 0
            assert not np.allclose(scores_3m[common].values, scores_6m[common].values)
