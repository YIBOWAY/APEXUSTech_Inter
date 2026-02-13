"""Unit tests for metrics_calculator.py.

Tests calculate_metrics() and build_equity_curve() with
hand-computable inputs so expected values are verifiable.
"""

import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.services.metrics_calculator import calculate_metrics, build_equity_curve


# ── calculate_metrics ────────────────────────────────────────────────


class TestCalculateMetrics:
    """Core metrics calculation tests."""

    def test_empty_returns(self):
        """Empty series should return all-zero metrics."""
        result = calculate_metrics(pd.Series(dtype=float))
        assert result["total_return"] == 0.0
        assert result["sharpe_ratio"] == 0.0
        assert result["total_months"] == 0

    def test_total_return(self, known_monthly_returns):
        """Total return = product of (1+r) - 1."""
        result = calculate_metrics(known_monthly_returns)
        expected = float(np.prod(1 + known_monthly_returns.values) - 1)
        assert abs(result["total_return"] - expected) < 1e-5

    def test_annual_return_geometric(self, known_monthly_returns):
        """Annual return should use geometric formula: (1+total)^(12/n) - 1."""
        result = calculate_metrics(known_monthly_returns)
        total = np.prod(1 + known_monthly_returns.values) - 1
        n = len(known_monthly_returns)
        expected = (1 + total) ** (12 / n) - 1
        assert abs(result["annual_return"] - round(expected, 6)) < 1e-5

    def test_annual_return_is_not_arithmetic(self, known_monthly_returns):
        """Verify we're NOT using the old arithmetic formula."""
        result = calculate_metrics(known_monthly_returns)
        arithmetic_annual = (1 + known_monthly_returns.mean()) ** 12 - 1
        # They should differ (geometric != arithmetic unless all returns identical)
        assert abs(result["annual_return"] - arithmetic_annual) > 1e-6

    def test_sharpe_with_risk_free_rate(self, known_monthly_returns):
        """Sharpe = (annual_return - rf) / annual_vol."""
        rf = 0.05  # 5% risk-free rate
        result = calculate_metrics(known_monthly_returns, annual_risk_free_rate=rf)

        total = np.prod(1 + known_monthly_returns.values) - 1
        n = len(known_monthly_returns)
        geo_annual = (1 + total) ** (12 / n) - 1
        annual_vol = known_monthly_returns.std() * np.sqrt(12)
        expected_sharpe = (geo_annual - rf) / annual_vol

        assert abs(result["sharpe_ratio"] - round(expected_sharpe, 4)) < 1e-3

    def test_sharpe_without_risk_free_defaults_to_zero(self, known_monthly_returns):
        """When no rf provided, Sharpe = annual_return / annual_vol."""
        result = calculate_metrics(known_monthly_returns)
        total = np.prod(1 + known_monthly_returns.values) - 1
        n = len(known_monthly_returns)
        geo_annual = (1 + total) ** (12 / n) - 1
        annual_vol = known_monthly_returns.std() * np.sqrt(12)
        expected_sharpe = geo_annual / annual_vol

        assert abs(result["sharpe_ratio"] - round(expected_sharpe, 4)) < 1e-3

    def test_sharpe_higher_rf_means_lower_sharpe(self, known_monthly_returns):
        """Higher risk-free rate should produce lower Sharpe."""
        low_rf = calculate_metrics(known_monthly_returns, annual_risk_free_rate=0.01)
        high_rf = calculate_metrics(known_monthly_returns, annual_risk_free_rate=0.10)
        assert low_rf["sharpe_ratio"] > high_rf["sharpe_ratio"]

    def test_max_drawdown(self):
        """Max drawdown with a known sequence: +10%, -20%, +5% → MDD = -20%."""
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        returns = pd.Series([0.10, -0.20, 0.05], index=dates)
        result = calculate_metrics(returns)

        # Cumulative: [1.10, 0.88, 0.924]
        # Peak:       [1.10, 1.10, 1.10]
        # Drawdown:   [0.0, -0.20, -0.16]
        assert abs(result["max_drawdown"] - (-0.20)) < 1e-4

    def test_win_rate(self, known_monthly_returns):
        """Win rate = fraction of positive months."""
        result = calculate_metrics(known_monthly_returns)
        positive_months = sum(1 for r in known_monthly_returns if r > 0)
        expected = positive_months / len(known_monthly_returns)
        assert abs(result["win_rate"] - expected) < 1e-4

    def test_profit_factor(self, known_monthly_returns):
        """Profit factor = sum(gains) / |sum(losses)|."""
        result = calculate_metrics(known_monthly_returns)
        gains = sum(r for r in known_monthly_returns if r > 0)
        losses = abs(sum(r for r in known_monthly_returns if r < 0))
        expected = gains / losses
        assert abs(result["profit_factor"] - round(expected, 4)) < 1e-3

    def test_profit_factor_no_losses(self):
        """All positive returns → profit_factor = 999.0."""
        dates = pd.date_range("2020-01-31", periods=3, freq="ME")
        returns = pd.Series([0.05, 0.03, 0.02], index=dates)
        result = calculate_metrics(returns)
        assert result["profit_factor"] == 999.0

    def test_total_months(self, known_monthly_returns):
        result = calculate_metrics(known_monthly_returns)
        assert result["total_months"] == 12

    def test_zero_volatility(self):
        """All identical returns → vol=0, sharpe=0 (no division by zero)."""
        dates = pd.date_range("2020-01-31", periods=5, freq="ME")
        returns = pd.Series([0.01, 0.01, 0.01, 0.01, 0.01], index=dates)
        result = calculate_metrics(returns)
        assert result["sharpe_ratio"] == 0.0
        assert result["annual_volatility"] == 0.0


# ── build_equity_curve ───────────────────────────────────────────────


class TestBuildEquityCurve:
    """Equity curve generation tests."""

    def test_empty_returns(self):
        result = build_equity_curve(pd.Series(dtype=float))
        assert result == []

    def test_start_value(self, known_monthly_returns):
        """First point should reflect first month's return applied to start_value."""
        curve = build_equity_curve(known_monthly_returns, start_value=10000.0)
        assert len(curve) == 12
        # First point: 10000 * (1 + 0.05) = 10500
        assert abs(curve[0]["value"] - 10500.0) < 0.01

    def test_final_value_matches_total_return(self, known_monthly_returns):
        """Final equity value should equal start * (1 + total_return)."""
        start = 10000.0
        curve = build_equity_curve(known_monthly_returns, start_value=start)
        total = np.prod(1 + known_monthly_returns.values)
        expected_final = start * total
        assert abs(curve[-1]["value"] - expected_final) < 0.1

    def test_date_format(self, known_monthly_returns):
        """Dates should be ISO format strings."""
        curve = build_equity_curve(known_monthly_returns)
        for pt in curve:
            assert len(pt["date"]) == 10  # "YYYY-MM-DD"
            assert pt["date"][4] == "-"

    def test_benchmark_normalization(self, known_monthly_returns):
        """Benchmark curve should start at start_value."""
        benchmark = pd.Series(
            [200, 210, 205, 215, 220, 218, 225, 230, 228, 235, 240, 245],
            index=known_monthly_returns.index,
        )
        curve = build_equity_curve(known_monthly_returns, benchmark, start_value=10000.0)
        # Benchmark starts at 200, normalized to 10000
        assert abs(curve[0]["benchmark"] - 10000.0 * (200 / 200)) < 0.01
        # Last point: 245 / 200 * 10000 = 12250
        assert abs(curve[-1]["benchmark"] - 12250.0) < 0.01

    def test_no_benchmark(self, known_monthly_returns):
        """Without benchmark, benchmark values should all be start_value."""
        curve = build_equity_curve(known_monthly_returns, start_value=10000.0)
        for pt in curve:
            assert pt["benchmark"] == 10000.0
