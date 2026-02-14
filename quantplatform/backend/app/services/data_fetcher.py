"""Yahoo Finance data fetcher with PostgreSQL caching.

Uses direct Yahoo Finance API requests (not yfinance library).
"""

import asyncio
import logging
from datetime import date, datetime

import pandas as pd
import requests
from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.price_data import DailyPrice

logger = logging.getLogger(__name__)

# HTTP headers for Yahoo Finance API
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def _fetch_yahoo_single(
    ticker: str, start_date: date, end_date: date
) -> tuple[pd.Series | None, pd.Series | None]:
    """Fetch one ticker from Yahoo Finance API using direct HTTP requests.

    Returns (adj_close_series, volume_series) or (None, None) on failure.
    """
    # Convert date to datetime for timestamp calculation
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d"

    try:
        response = requests.get(url, headers=YAHOO_HEADERS, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
            result = data["chart"]["result"][0]
            timestamps = result.get("timestamp", [])

            if timestamps and len(timestamps) > 10:
                quote = result["indicators"]["quote"][0]
                adjclose_data = result["indicators"].get("adjclose", [{}])
                adjclose = (
                    adjclose_data[0].get("adjclose", quote.get("close", []))
                    if adjclose_data
                    else quote.get("close", [])
                )
                vol = quote.get("volume", [])

                # Create date index (normalized to remove time component)
                dates = pd.to_datetime(timestamps, unit="s").normalize()

                # Create Series
                adj_close_series = pd.Series(adjclose, index=dates, name=ticker)
                volume_series = pd.Series(vol, index=dates, name=ticker)

                # Ensure timezone-naive
                if adj_close_series.index.tz is not None:
                    adj_close_series.index = adj_close_series.index.tz_localize(None)
                if volume_series.index.tz is not None:
                    volume_series.index = volume_series.index.tz_localize(None)

                # Remove duplicates
                adj_close_series = adj_close_series[
                    ~adj_close_series.index.duplicated(keep="first")
                ]
                volume_series = volume_series[
                    ~volume_series.index.duplicated(keep="first")
                ]

                return adj_close_series, volume_series

        logger.warning(f"No valid data returned for {ticker}")
        return None, None

    except Exception as e:
        logger.error(f"Failed to fetch {ticker}: {e}")
        return None, None


async def _load_from_db(
    db: AsyncSession, tickers: list[str], start_date: date, end_date: date
) -> dict[str, list[DailyPrice]]:
    """Load cached prices from database, grouped by ticker."""
    stmt = (
        select(DailyPrice)
        .where(
            and_(
                DailyPrice.ticker.in_(tickers),
                DailyPrice.trade_date >= start_date,
                DailyPrice.trade_date <= end_date,
            )
        )
        .order_by(DailyPrice.trade_date)
    )
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
            "adj_close": (
                float(adj_close[dt]) if pd.notna(adj_close[dt]) else None
            ),
            "volume": (
                int(volume[dt])
                if volume is not None
                and dt in volume.index
                and pd.notna(volume[dt])
                else None
            ),
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

    # Check which tickers have sufficient cached data covering the requested range
    for ticker in tickers:
        if ticker in cached and len(cached[ticker]) > 10:
            rows = cached[ticker]
            earliest_cached = min(r.trade_date for r in rows)
            # If cached data starts >30 days after requested start,
            # we're missing early data — need to re-fetch the full range.
            # Yahoo returns the complete range; ON CONFLICT DO NOTHING
            # skips dates already in DB.
            if (earliest_cached - start_date).days > 30:
                tickers_to_fetch.append(ticker)
                logger.info(
                    f"  {ticker}: cache starts at {earliest_cached}, "
                    f"but requested {start_date} — will re-fetch"
                )
            else:
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
            if adj_s is not None and len(adj_s) > 10:
                adj_close_dict[ticker] = adj_s
                volume_dict[ticker] = vol_s
                await _save_to_db(db, ticker, adj_s, vol_s)
                logger.info(f"  {ticker}: {len(adj_s)} rows fetched and cached")
            else:
                logger.warning(f"  {ticker}: fetch failed, skipping")

            # Small delay to be respectful to Yahoo's servers
            await asyncio.sleep(0.3)

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
