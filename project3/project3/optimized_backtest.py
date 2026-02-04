"""
Project3 wrapper for optimized backtest framework.

Re-exports classes/functions from the root-level optimized_backtest.py
so notebooks can import from project3.optimized_backtest.
"""

from optimized_backtest import (
    Event,
    MarketEvent,
    SignalEvent,
    OrderEvent,
    FillEvent,
    OptimizedCSVDataHandler,
    OptimizedCalendarSpreadZScoreStrategy,
    OptimizedBasicPortfolio,
    OptimizedSimulatedExecutionHandler,
    OptimizedBacktest,
    benchmark_backtest,
)

__all__ = [
    "Event",
    "MarketEvent",
    "SignalEvent",
    "OrderEvent",
    "FillEvent",
    "OptimizedCSVDataHandler",
    "OptimizedCalendarSpreadZScoreStrategy",
    "OptimizedBasicPortfolio",
    "OptimizedSimulatedExecutionHandler",
    "OptimizedBacktest",
    "benchmark_backtest",
]
