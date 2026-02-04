from pydantic import BaseModel


class PricePoint(BaseModel):
    date: str
    adj_close: float | None
    volume: int | None


class FetchRequest(BaseModel):
    tickers: list[str]
    start_date: str
    end_date: str
