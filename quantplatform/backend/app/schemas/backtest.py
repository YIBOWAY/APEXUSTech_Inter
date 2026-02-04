from pydantic import BaseModel


class BacktestRequest(BaseModel):
    start_date: str = "2020-01-01"
    end_date: str = "2026-01-30"
    method: str | None = None
    lookback_months: int | None = None


class EquityPointOut(BaseModel):
    date: str
    value: float
    benchmark: float


class TradeOut(BaseModel):
    id: str
    symbol: str
    direction: str
    price: float
    quantity: float
    timestamp: str
    commission: float
    pnl: float | None
    status: str


class BacktestRunOut(BaseModel):
    id: str
    strategy_id: str
    status: str
    started_at: str | None = None
    finished_at: str | None = None
    total_return: float | None = None
    annual_return: float | None = None
    annual_volatility: float | None = None
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    win_rate: float | None = None
    profit_factor: float | None = None
    total_months: int | None = None
    error_message: str | None = None

    model_config = {"from_attributes": True}
