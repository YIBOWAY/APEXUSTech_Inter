"""
Optimized Event-Driven Backtesting Framework (v3)

Key v3 additions:
1) Next-bar execution: signals at t, fills at t+1
2) Risk rules: cash sufficiency, max leverage/position, max drawdown halt

This file is v3-only and does not change the v2 engine.
"""

from __future__ import annotations

import queue
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd


# ==================== Event Classes ====================

class Event:
    """Base Event class"""
    pass


class MarketEvent(Event):
    """Handles the event of receiving a new market update with corresponding bars."""

    def __init__(self):
        self.type = "MARKET"


class SignalEvent(Event):
    """Handles the event of sending a Signal from a Strategy object."""

    def __init__(self, symbol: str, datetime: datetime, signal_type: str):
        self.type = "SIGNAL"
        self.symbol = symbol
        self.datetime = datetime
        self.signal_type = signal_type  # LONG_SPREAD, SHORT_SPREAD, EXIT_LONG, EXIT_SHORT


class OrderEvent(Event):
    """Handles the event of sending an Order to an execution system."""

    def __init__(self, symbol: str, order_type: str, quantity: int, direction: str):
        self.type = "ORDER"
        self.symbol = symbol
        self.order_type = order_type  # MKT, LMT
        self.quantity = quantity
        self.direction = direction  # BUY, SELL

    def __repr__(self):
        return (
            "OrderEvent: "
            f"Symbol={self.symbol}, Type={self.order_type}, "
            f"Quantity={self.quantity}, Direction={self.direction}"
        )


class FillEvent(Event):
    """Encapsulates the notion of a Filled Order."""

    def __init__(self, timeindex: datetime, symbol: str, quantity: int,
                 direction: str, fill_cost: float, commission: float = None):
        self.type = "FILL"
        self.timeindex = timeindex
        self.symbol = symbol
        self.quantity = quantity
        self.direction = direction
        self.fill_cost = fill_cost
        self.commission = self.calculate_commission() if commission is None else commission

    def calculate_commission(self) -> float:
        return max(1.0, 0.001 * self.quantity)


# ==================== Data Handler ====================

class OptimizedCSVDataHandlerV3:
    """
    Optimized CSV Data Handler using numpy arrays instead of recarray.
    """

    def __init__(self, events_queue: queue.Queue, csv_path: str):
        self.events = events_queue
        self.csv_path = csv_path
        self.symbol_data: Dict[str, np.ndarray] = {}
        self.symbols: List[str] = []
        self.current_idx = 0
        self.total_bars = 0
        self.dates: Optional[np.ndarray] = None
        self.continue_backtest = True
        self.latest_bar: Optional[Dict[str, Any]] = None

        self._open_convert_csv_files()

    def _open_convert_csv_files(self):
        df = pd.read_csv(self.csv_path, header=0, index_col=0, parse_dates=True)
        self.dates = df.index.to_numpy()
        self.symbol_data = {col: df[col].to_numpy() for col in df.columns}
        self.symbols = list(df.columns)
        self.total_bars = len(self.dates)

    def update_bars(self):
        if self.current_idx >= self.total_bars:
            self.continue_backtest = False
            return

        self.latest_bar = {
            "Date": self.dates[self.current_idx],
            **{symbol: self.symbol_data[symbol][self.current_idx] for symbol in self.symbols},
        }
        self.current_idx += 1
        self.events.put(MarketEvent())

    def get_latest_bar(self, symbol: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return self.latest_bar


# ==================== Strategy ====================

class OptimizedCalendarSpreadZScoreStrategyV3:
    """Optimized Calendar Spread Z-Score Strategy using deque for O(1) rolling stats."""

    def __init__(self, data_handler: OptimizedCSVDataHandlerV3, events_queue: queue.Queue,
                 symbol: str, lookback_window: int = 60, z_threshold: float = 2.0):
        self.data_handler = data_handler
        self.events = events_queue
        self.symbol = symbol
        self.lookback_window = lookback_window
        self.z_threshold = z_threshold

        self.spread_window: deque = deque(maxlen=lookback_window)
        self.spread_sum = 0.0
        self.spread_sum_sq = 0.0
        self._date_field: Optional[str] = None
        self.bought = False
        self.sold = False
        self.z_scores: Dict[datetime, float] = {}

    def _get_bar_date(self, bar: Dict[str, Any]) -> datetime:
        if self._date_field is None:
            self._date_field = "Date" if "Date" in bar else list(bar.keys())[0]
        return bar[self._date_field]

    def calculate_signals(self, event: MarketEvent):
        if event.type != "MARKET":
            return
        bar = self.data_handler.get_latest_bar(self.symbol)
        if bar is None:
            return

        spread = bar.get("FAR", 0) - bar.get("NEAR", 0)
        bar_date = self._get_bar_date(bar)

        if len(self.spread_window) == self.lookback_window:
            old_spread = self.spread_window[0]
            self.spread_sum -= old_spread
            self.spread_sum_sq -= old_spread * old_spread

        self.spread_window.append(spread)
        self.spread_sum += spread
        self.spread_sum_sq += spread * spread

        n = len(self.spread_window)
        if n < self.lookback_window:
            return

        rolling_mean = self.spread_sum / n
        mean_sq = (self.spread_sum / n) ** 2
        variance = (self.spread_sum_sq / n) - mean_sq
        rolling_std = np.sqrt(max(0, variance))
        if rolling_std <= 0:
            return

        z_score = (spread - rolling_mean) / rolling_std
        self.z_scores[bar_date] = z_score

        if not self.sold and z_score > self.z_threshold:
            signal = SignalEvent(self.symbol, bar_date, "SHORT_SPREAD")
            self.events.put(signal)
            self.sold = True
        elif not self.bought and z_score < -self.z_threshold:
            signal = SignalEvent(self.symbol, bar_date, "LONG_SPREAD")
            self.events.put(signal)
            self.bought = True
        elif self.sold and z_score <= 0.5:
            signal = SignalEvent(self.symbol, bar_date, "EXIT_SHORT")
            self.events.put(signal)
            self.sold = False
        elif self.bought and z_score >= -0.5:
            signal = SignalEvent(self.symbol, bar_date, "EXIT_LONG")
            self.events.put(signal)
            self.bought = False


# ==================== Risk Model ====================

@dataclass
class RiskConfig:
    max_leverage: float = 2.0
    max_position_per_leg: int = 20
    max_drawdown_pct: float = 0.10


class RiskManager:
    def __init__(self, risk_config: RiskConfig):
        self.risk_config = risk_config

    def allow_order(self, cash: float, est_cost: float) -> bool:
        return cash >= est_cost

    def allow_leverage(self, equity: float, gross_exposure: float) -> bool:
        if equity <= 0:
            return False
        return gross_exposure <= equity * self.risk_config.max_leverage

    def allow_drawdown(self, equity_curve: pd.DataFrame) -> bool:
        if equity_curve.empty:
            return True
        running_max = equity_curve["total"].cummax()
        dd = (equity_curve["total"] / running_max - 1.0).min()
        return dd >= -self.risk_config.max_drawdown_pct


# ==================== Portfolio ====================

class OptimizedBasicPortfolioV3:
    def __init__(self, data_handler: OptimizedCSVDataHandlerV3, events_queue: queue.Queue,
                 initial_capital: float = 100000.0, quantity: int = 10):
        self.data_handler = data_handler
        self.events = events_queue
        self.initial_capital = initial_capital
        self.quantity = quantity

        self.current_positions: Dict[str, int] = {symbol: 0 for symbol in self.data_handler.symbols}
        self.current_holdings = {
            "datetime": None,
            "cash": self.initial_capital,
            "commission": 0.0,
            "total": self.initial_capital,
        }
        self.all_holdings: List[Dict[str, Any]] = []

    def update_timeindex(self, event: MarketEvent):
        if event.type != "MARKET":
            return
        bar = self.data_handler.get_latest_bar()
        if bar is None:
            return
        position_value = sum(
            self.current_positions[symbol] * bar.get(symbol, 0)
            for symbol in self.data_handler.symbols
        )
        total = self.current_holdings["cash"] + position_value
        self.all_holdings.append({
            "datetime": bar.get("Date"),
            "cash": self.current_holdings["cash"],
            "commission": self.current_holdings["commission"],
            "total": total,
        })
        self.current_holdings["total"] = total

    def update_positions_from_fill(self, event: FillEvent):
        fill_dir = 1 if event.direction == "BUY" else -1
        self.current_positions[event.symbol] += fill_dir * event.quantity

    def update_holdings_from_fill(self, event: FillEvent):
        fill_dir = 1 if event.direction == "BUY" else -1
        fill_cost = event.fill_cost * event.quantity
        cost = fill_dir * fill_cost
        self.current_holdings["commission"] += event.commission
        self.current_holdings["cash"] -= (cost + event.commission)
        self.current_holdings["total"] -= event.commission

    def generate_naive_order(self, event: SignalEvent) -> List[OrderEvent]:
        if event.type != "SIGNAL":
            return []
        orders: List[OrderEvent] = []
        if event.signal_type == "LONG_SPREAD":
            orders = [
                OrderEvent("FAR", "MKT", self.quantity, "BUY"),
                OrderEvent("NEAR", "MKT", self.quantity, "SELL"),
            ]
        elif event.signal_type == "SHORT_SPREAD":
            orders = [
                OrderEvent("FAR", "MKT", self.quantity, "SELL"),
                OrderEvent("NEAR", "MKT", self.quantity, "BUY"),
            ]
        elif event.signal_type == "EXIT_SHORT":
            if self.current_positions.get("FAR", 0) < 0:
                orders = [
                    OrderEvent("FAR", "MKT", abs(self.current_positions["FAR"]), "BUY"),
                    OrderEvent("NEAR", "MKT", self.current_positions.get("NEAR", 0), "SELL"),
                ]
        elif event.signal_type == "EXIT_LONG":
            if self.current_positions.get("FAR", 0) > 0:
                orders = [
                    OrderEvent("FAR", "MKT", self.current_positions["FAR"], "SELL"),
                    OrderEvent("NEAR", "MKT", abs(self.current_positions.get("NEAR", 0)), "BUY"),
                ]
        return orders

    def create_equity_curve_dataframe(self) -> pd.DataFrame:
        if not self.all_holdings:
            return pd.DataFrame()
        curve = pd.DataFrame(self.all_holdings)
        curve.set_index("datetime", inplace=True)
        curve["returns"] = curve["total"].pct_change()
        curve["equity_curve"] = (1.0 + curve["returns"]).cumprod()
        return curve


# ==================== Execution Handler ====================

class OptimizedSimulatedExecutionHandlerV3:
    def __init__(self, events_queue: queue.Queue, data_handler: OptimizedCSVDataHandlerV3,
                 commission_per_trade: float = 5.0, slippage_per_trade: float = 0.01):
        self.events = events_queue
        self.data_handler = data_handler
        self.commission = commission_per_trade
        self.slippage = slippage_per_trade
        self._date_accessor: Optional[str] = None

    def _get_bar_date(self, bar: Dict[str, Any]) -> datetime:
        if self._date_accessor is None:
            self._date_accessor = "Date" if "Date" in bar else list(bar.keys())[0]
        return bar[self._date_accessor]

    def execute_order(self, event: OrderEvent):
        if event.type != "ORDER":
            return
        bar = self.data_handler.get_latest_bar()
        if bar is None:
            return

        base_price = bar.get(event.symbol, 0)
        if event.direction == "BUY":
            fill_price = base_price + self.slippage
        else:
            fill_price = base_price - self.slippage

        fill_event = FillEvent(
            timeindex=self._get_bar_date(bar),
            symbol=event.symbol,
            quantity=event.quantity,
            direction=event.direction,
            fill_cost=fill_price,
            commission=self.commission,
        )
        self.events.put(fill_event)


# ==================== Backtest Engine (v3) ====================

class OptimizedBacktestV3:
    def __init__(self, csv_path: str, symbol: str, initial_capital: float = 100000.0,
                 quantity: int = 10, lookback_window: int = 60, z_threshold: float = 2.0,
                 commission: float = 5.0, slippage: float = 0.01,
                 execution_delay: int = 1, risk_config: Optional[RiskConfig] = None):
        self.csv_path = csv_path
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.quantity = quantity
        self.lookback_window = lookback_window
        self.z_threshold = z_threshold
        self.commission = commission
        self.slippage = slippage
        self.execution_delay = max(1, execution_delay)
        self.risk_config = risk_config or RiskConfig(
            max_leverage=2.0,
            max_position_per_leg=quantity * 2,
            max_drawdown_pct=0.10,
        )

        self.events = queue.Queue()
        self.data_handler = OptimizedCSVDataHandlerV3(self.events, self.csv_path)
        self.strategy = OptimizedCalendarSpreadZScoreStrategyV3(
            self.data_handler, self.events, self.symbol, self.lookback_window, self.z_threshold
        )
        self.portfolio = OptimizedBasicPortfolioV3(
            self.data_handler, self.events, self.initial_capital, self.quantity
        )
        self.execution_handler = OptimizedSimulatedExecutionHandlerV3(
            self.events, self.data_handler, self.commission, self.slippage
        )
        self.risk_manager = RiskManager(self.risk_config)
        self._pending_orders: deque = deque()

    def _queue_orders_with_delay(self, orders: List[OrderEvent]):
        for order in orders:
            self._pending_orders.append({"delay": self.execution_delay, "order": order})

    def _process_pending_orders(self):
        ready: List[OrderEvent] = []
        for _ in range(len(self._pending_orders)):
            item = self._pending_orders.popleft()
            item["delay"] -= 1
            if item["delay"] <= 0:
                ready.append(item["order"])
            else:
                self._pending_orders.append(item)
        for order in ready:
            self.events.put(order)

    def _estimate_order_cost(self, order: OrderEvent, bar: Dict[str, Any]) -> float:
        price = bar.get(order.symbol, 0)
        if order.direction == "BUY":
            price += self.slippage
        else:
            price -= self.slippage
        return price * order.quantity + self.commission

    def _gross_exposure(self, bar: Dict[str, Any]) -> float:
        return sum(
            abs(self.portfolio.current_positions[symbol]) * bar.get(symbol, 0)
            for symbol in self.data_handler.symbols
        )

    def _run_backtest(self):
        print("Running optimized backtest v3...")
        while True:
            self.data_handler.update_bars()
            if not self.data_handler.continue_backtest:
                break

            # Next-bar execution: first process any pending orders whose delay elapsed
            self._process_pending_orders()

            while not self.events.empty():
                event = self.events.get()
                if event is None:
                    continue

                if event.type == "MARKET":
                    self.portfolio.update_timeindex(event)
                    self.strategy.calculate_signals(event)
                elif event.type == "SIGNAL":
                    bar = self.data_handler.get_latest_bar()
                    if bar is None:
                        continue

                    # Risk rule D: max drawdown
                    curve = self.portfolio.create_equity_curve_dataframe()
                    if not self.risk_manager.allow_drawdown(curve):
                        continue

                    orders = self.portfolio.generate_naive_order(event)
                    if not orders:
                        continue

                    # Risk rule B: max position per leg & leverage
                    gross_exposure = self._gross_exposure(bar)
                    equity = self.portfolio.current_holdings.get("total", 0.0)
                    if not self.risk_manager.allow_leverage(equity, gross_exposure):
                        continue
                    if any(abs(self.portfolio.current_positions.get(o.symbol, 0)) >
                           self.risk_config.max_position_per_leg for o in orders):
                        continue

                    # Risk rule A: cash sufficiency (estimate total cost)
                    est_cost = sum(self._estimate_order_cost(o, bar) for o in orders)
                    if not self.risk_manager.allow_order(
                        self.portfolio.current_holdings.get("cash", 0.0), est_cost
                    ):
                        continue

                    self._queue_orders_with_delay(orders)
                elif event.type == "ORDER":
                    self.execution_handler.execute_order(event)
                elif event.type == "FILL":
                    self.portfolio.update_positions_from_fill(event)
                    self.portfolio.update_holdings_from_fill(event)

    def simulate_trading(self) -> Dict[str, Any]:
        self._run_backtest()
        curve = self.portfolio.create_equity_curve_dataframe()
        if curve.empty:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "volatility": 0.0,
                "equity_curve": curve,
            }
        returns = curve["returns"].dropna()
        if len(returns) == 0:
            return {
                "sharpe_ratio": 0.0,
                "max_drawdown": 0.0,
                "total_return": 0.0,
                "volatility": 0.0,
                "equity_curve": curve,
            }
        sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std()) if returns.std() != 0 else 0.0
        max_drawdown = (curve["total"] / curve["total"].cummax() - 1.0).min()
        total_return = curve["total"].iloc[-1] / curve["total"].iloc[0] - 1.0
        volatility = returns.std() * np.sqrt(252)
        return {
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "total_return": total_return,
            "volatility": volatility,
            "equity_curve": curve,
        }
