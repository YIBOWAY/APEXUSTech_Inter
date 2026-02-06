"""Market data API endpoints."""

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.price_data import DailyPrice
from app.schemas.market_data import PricePoint, FetchRequest
from app.services.data_fetcher import fetch_and_cache_prices

router = APIRouter(prefix="/market-data", tags=["market-data"])


@router.get("/{ticker}", response_model=list[PricePoint])
async def get_prices(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Get cached price data for a ticker."""
    stmt = select(DailyPrice).where(DailyPrice.ticker == ticker)

    if start_date:
        stmt = stmt.where(DailyPrice.trade_date >= datetime.strptime(start_date, "%Y-%m-%d").date())
    if end_date:
        stmt = stmt.where(DailyPrice.trade_date <= datetime.strptime(end_date, "%Y-%m-%d").date())

    stmt = stmt.order_by(DailyPrice.trade_date)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    return [
        PricePoint(
            date=row.trade_date.isoformat(),
            adj_close=row.adj_close,
            volume=row.volume,
        )
        for row in rows
    ]


@router.post("/fetch")
async def fetch_prices(request: FetchRequest, db: AsyncSession = Depends(get_db)):
    """Manually trigger data fetch for specified tickers."""
    start_dt = datetime.strptime(request.start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(request.end_date, "%Y-%m-%d").date()

    adj_close, volume = await fetch_and_cache_prices(
        db, request.tickers, start_dt, end_dt
    )

    return {
        "fetched": len(adj_close.columns) if not adj_close.empty else 0,
        "start_date": request.start_date,
        "end_date": request.end_date,
    }
