"""Unit tests for backtest_strategy() and run_full_backtest() in momentum_strategy.py.

Uses synthetic 5-stock data to verify the full backtest pipeline:
monthly rebalancing, trade signal generation, transaction costs, and output format.
"""

import numpy as np
import pandas as pd
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "backend"))

from app.services.momentum_strategy import backtest_strategy, run_full_backtest


class TestBacktestStrategy:
    """Tests for the core backtest_strategy() function."""

    def test_returns_series_not_empty(self, simple_adj_close, simple_volume):
        """Should produce a non-empty return series."""
        returns, signals = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.4,
        )
        assert not returns.empty
        assert len(returns) > 0

    def test_returns_are_floats(self, simple_adj_close, simple_volume):
        """Monthly returns should all be finite floats."""
        returns, _ = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.4,
        )
        assert all(np.isfinite(r) for r in returns)

    def test_trade_signals_structure(self, simple_adj_close, simple_volume):
        """Trade signals should have required keys."""
        _, signals = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.4,
        )
        assert len(signals) > 0
        required_keys = {"signal_date", "ticker", "direction", "weight", "score", "price"}
        for sig in signals:
            assert required_keys.issubset(sig.keys()), f"Missing keys in signal: {sig.keys()}"

    def test_signals_direction_is_buy(self, simple_adj_close, simple_volume):
        """All signals should be BUY (long-only strategy)."""
        _, signals = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.4,
        )
        for sig in signals:
            assert sig["direction"] == "BUY"

    def test_weights_sum_to_one_per_month(self, simple_adj_close, simple_volume):
        """Weights for each signal date should sum to ~1.0 (equal-weighted)."""
        _, signals = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.4,
        )
        from collections import defaultdict
        date_weights = defaultdict(float)
        for sig in signals:
            date_weights[sig["signal_date"]] += sig["weight"]

        for date_str, total_weight in date_weights.items():
            assert abs(total_weight - 1.0) < 0.01, f"Weights for {date_str} sum to {total_weight}"

    def test_transaction_cost_reduces_returns(self, simple_adj_close, simple_volume):
        """Higher TC should produce lower returns (on average)."""
        returns_low_tc, _ = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.4, tc_bps=0,
        )
        returns_high_tc, _ = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.4, tc_bps=50,
        )
        # With same random seed, high TC should drag returns down
        assert returns_low_tc.sum() >= returns_high_tc.sum()

    def test_different_methods_produce_different_signals(self, simple_adj_close, simple_volume):
        """Simple vs risk_adjusted should select different stocks at least once."""
        _, signals_simple = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.2,
        )
        _, signals_risk = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="risk_adjusted", decile=0.2,
        )
        # Collect tickers selected by each method per date
        def tickers_by_date(signals):
            result = {}
            for s in signals:
                result.setdefault(s["signal_date"], set()).add(s["ticker"])
            return result

        simple_sel = tickers_by_date(signals_simple)
        risk_sel = tickers_by_date(signals_risk)
        common_dates = set(simple_sel) & set(risk_sel)

        # At least one month should have a different stock selection
        any_diff = any(simple_sel[d] != risk_sel[d] for d in common_dates)
        assert any_diff or len(common_dates) == 0, "Methods selected identical portfolios every month"

    def test_decile_affects_portfolio_size(self, simple_adj_close, simple_volume):
        """Smaller decile → fewer stocks selected per month."""
        _, signals_wide = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.6,
        )
        _, signals_narrow = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.2,
        )
        # Average signals per month should differ
        if signals_wide and signals_narrow:
            from collections import Counter
            dates_wide = Counter(s["signal_date"] for s in signals_wide)
            dates_narrow = Counter(s["signal_date"] for s in signals_narrow)
            avg_wide = np.mean(list(dates_wide.values()))
            avg_narrow = np.mean(list(dates_narrow.values()))
            assert avg_wide >= avg_narrow

    def test_winsorization(self, simple_adj_close, simple_volume):
        """Strategy should handle winsorization without error."""
        returns, _ = backtest_strategy(
            simple_adj_close, simple_volume,
            lookback_months=6, method="simple", decile=0.4, winsor_q=0.05,
        )
        assert not returns.empty
        # No extreme outliers (within reasonable bounds)
        assert returns.max() < 1.0   # < 100% monthly return
        assert returns.min() > -1.0  # > -100% monthly return


class TestRunFullBacktest:
    """Tests for run_full_backtest() — the top-level orchestrator."""

    def test_output_structure(self, simple_adj_close, simple_volume):
        """Output dict should have all required keys."""
        result = run_full_backtest(
            simple_adj_close, simple_volume,
            benchmark_ticker="XLK", lookback_months=6,
            method="simple", decile=0.4,
        )
        assert "monthly_returns" in result
        assert "equity_curve" in result
        assert "metrics" in result
        assert "trade_signals" in result

    def test_metrics_keys(self, simple_adj_close, simple_volume):
        """Metrics dict should contain standard performance metrics."""
        result = run_full_backtest(
            simple_adj_close, simple_volume,
            benchmark_ticker="XLK", lookback_months=6,
            method="simple", decile=0.4,
        )
        metrics = result["metrics"]
        expected_keys = {
            "total_return", "annual_return", "annual_volatility",
            "sharpe_ratio", "max_drawdown", "win_rate",
            "profit_factor", "total_months",
        }
        assert expected_keys.issubset(metrics.keys())

    def test_equity_curve_format(self, simple_adj_close, simple_volume):
        """Each equity point should have date, value, benchmark."""
        result = run_full_backtest(
            simple_adj_close, simple_volume,
            benchmark_ticker="XLK", lookback_months=6,
            method="simple", decile=0.4,
        )
        if result["equity_curve"]:
            pt = result["equity_curve"][0]
            assert "date" in pt
            assert "value" in pt
            assert "benchmark" in pt

    def test_monthly_returns_format(self, simple_adj_close, simple_volume):
        """Each monthly return should have date and return."""
        result = run_full_backtest(
            simple_adj_close, simple_volume,
            benchmark_ticker="XLK", lookback_months=6,
            method="simple", decile=0.4,
        )
        if result["monthly_returns"]:
            mr = result["monthly_returns"][0]
            assert "date" in mr
            assert "return" in mr

    def test_risk_free_rate_from_irx(self, simple_adj_close, simple_volume):
        """When ^IRX is in data, Sharpe should incorporate risk-free rate.

        Compare Sharpe with ^IRX present vs absent — they should differ.
        """
        # With ^IRX
        result_with_rf = run_full_backtest(
            simple_adj_close, simple_volume,
            benchmark_ticker="XLK", lookback_months=6,
            method="simple", decile=0.4,
        )

        # Without ^IRX
        adj_no_irx = simple_adj_close.drop(columns=["^IRX"])
        result_without_rf = run_full_backtest(
            adj_no_irx, simple_volume,
            benchmark_ticker="XLK", lookback_months=6,
            method="simple", decile=0.4,
        )

        sharpe_with = result_with_rf["metrics"].get("sharpe_ratio", 0)
        sharpe_without = result_without_rf["metrics"].get("sharpe_ratio", 0)

        # With positive risk-free rate, Sharpe should be lower
        assert sharpe_with < sharpe_without or sharpe_with == sharpe_without == 0

    def test_empty_data(self):
        """Empty input (with DatetimeIndex) should return empty results."""
        empty_idx = pd.DatetimeIndex([], dtype="datetime64[ns]")
        result = run_full_backtest(
            pd.DataFrame(index=empty_idx), pd.DataFrame(index=empty_idx),
            benchmark_ticker="XLK", lookback_months=6,
        )
        assert result["monthly_returns"] == []
        assert result["equity_curve"] == []
        assert result["trade_signals"] == []
