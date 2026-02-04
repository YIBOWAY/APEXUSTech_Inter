"""Yahoo Finance data fetcher with PostgreSQL caching."""

import logging
from datetime import date, datetime

import numpy as np
import pandas as pd
import requests
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.price_data import DailyPrice

logger = logging.getLogger(__name__)


def _fetch_yahoo_single(
    ticker: str, start_date: date, end_date: date
) -> tuple[pd.Series | None, pd.Series | None]:
    """Fetch one ticker from Yahoo Finance HTTP API.

    Returns (adj_close_series, volume_series) or (None, None) on failure.
    """
    start_ts = int(datetime.combine(start_date, datetime.min.time()).timestamp())
    end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
        f"?period1={start_ts}&period2={end_ts}&interval=1d"
    )

    try:
        resp = requests.get(url, headers=settings.YAHOO_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]

        timestamps = result["timestamps"]
        if len(timestamps) < 10:
            logger.warning(f"Insufficient data for {ticker}: {len(timestamps)} rows")
            return None, None

        quote = result["indicators"]["quote"][0]
        adj_data = result["indicators"].get("adjclose", [{}])
        adj_close_vals = (
            adj_data[0].get("adjclose", quote["close"])
            if adj_data
            else quote["close"]
        )
        volume_vals = quote.get("volume", [])

        dates = pd.to_datetime(timestamps, unit="s").normalize().tz_localize(None)
        adj_close = pd.Series(adj_close_vals, index=dates, name=ticker, dtype=float)
        vol_series = pd.Series(volume_vals, index=dates, name=ticker, dtype=float) if volume_vals else None

        # Remove duplicates
        adj_close = adj_close[~adj_close.index.duplicated(keep="first")]
        if vol_series is not None:
            vol_series = vol_series[~vol_series.index.duplicated(keep="first")]

        return adj_close, vol_series
    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return None, None


async def _load_from_db(
    db: AsyncSession, tickers: list[str], start_date: date, end_date: date
) -> dict[str, list[DailyPrice]]:
    """Load cached prices from database, grouped by ticker."""
    stmt = select(DailyPrice).where(
        and_(
            DailyPrice.ticker.in_(tickers),
            DailyPrice.trade_date >= start_date,
            DailyPrice.trade_date <= end_date,
        )
    ).order_by(DailyPrice.trade_date)
    result = await db.execute(stmt)
    rows = result.scalars().all()

    grouped: dict[str, list[DailyPrice]] = {}
    for row in rows:
        grouped.setdefault(row.ticker, []).append(row)
    return grouped


async def _save_to_db(
    db: AsyncSession,
    ticker: str,
    adj_close: pd.Series,
    volume: pd.Series | None,
):
    """Upsert price data into daily_prices table."""
    records = []
    for dt in adj_close.index:
        rec = {
            "ticker": ticker,
            "trade_date": dt.date() if hasattr(dt, "date") else dt,
            "adj_close": float(adj_close[dt]) if pd.notna(adj_close[dt]) else None,
            "volume": int(volume[dt]) if volume is not None and dt in volume.index and pd.notna(volume[dt]) else None,
            "source": "yahoo",
        }
        records.append(rec)

    if not records:
        return

    stmt = pg_insert(DailyPrice).values(records)
    stmt = stmt.on_conflict_do_nothing(constraint="uq_ticker_date")
    await db.execute(stmt)
    await db.commit()


async def fetch_and_cache_prices(
    db: AsyncSession,
    tickers: list[str],
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch prices with DB caching. Returns (adj_close_df, volume_df).

    1. Check DB for cached data
    2. Fetch missing tickers from Yahoo API
    3. Save new data to DB
    4. Return combined DataFrames
    """
    cached = await _load_from_db(db, tickers, start_date, end_date)

    adj_close_dict: dict[str, pd.Series] = {}
    volume_dict: dict[str, pd.Series] = {}
    tickers_to_fetch: list[str] = []

    # Check which tickers have sufficient cached data
    for ticker in tickers:
        if ticker in cached and len(cached[ticker]) > 10:
            rows = cached[ticker]
            dates = pd.DatetimeIndex([r.trade_date for r in rows])
            adj_close_dict[ticker] = pd.Series(
                [r.adj_close for r in rows], index=dates, name=ticker, dtype=float
            )
            volume_dict[ticker] = pd.Series(
                [r.volume for r in rows], index=dates, name=ticker, dtype=float
            )
            logger.info(f"  {ticker}: {len(rows)} rows from cache")
        else:
            tickers_to_fetch.append(ticker)

    # Fetch missing from Yahoo API
    if tickers_to_fetch:
        logger.info(f"Fetching {len(tickers_to_fetch)} tickers from Yahoo API...")
        for ticker in tickers_to_fetch:
            adj_s, vol_s = _fetch_yahoo_single(ticker, start_date, end_date)
            if adj_s is not None:
                adj_close_dict[ticker] = adj_s
                volume_dict[ticker] = vol_s
                await _save_to_db(db, ticker, adj_s, vol_s)
                logger.info(f"  {ticker}: {len(adj_s)} rows fetched and cached")
            else:
                logger.warning(f"  {ticker}: fetch failed, skipping")

    # Build DataFrames
    if not adj_close_dict:
        return pd.DataFrame(), pd.DataFrame()

    adj_close_df = pd.DataFrame(adj_close_dict).sort_index()
    volume_df = pd.DataFrame(
        {k: v for k, v in volume_dict.items() if v is not None}
    ).sort_index()

    # Forward fill missing values (max 5 days)
    adj_close_df = adj_close_df.ffill(limit=5)
    volume_df = volume_df.ffill(limit=5)

    return adj_close_df, volume_df
