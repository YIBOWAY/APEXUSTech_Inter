from pydantic import BaseModel


class MetricsOut(BaseModel):
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    annual_volatility: float
    win_rate: float
    profit_factor: float


class StrategyOut(BaseModel):
    id: str
    name: str
    description: str
    tags: list[str]
    metrics: MetricsOut | None = None
    method: str
    lookback_months: int
    status: str
    created_at: str
    updated_at: str

    model_config = {"from_attributes": True}


class StrategyCreate(BaseModel):
    name: str
    description: str = ""
    tags: list[str] = []
    method: str = "simple"
    lookback_months: int = 6
    decile: float = 0.2
    tc_bps: int = 5
    winsor_q: float = 0.01
