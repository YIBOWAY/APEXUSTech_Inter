"""Simple test script for data_fetcher Yahoo Finance API.

This script tests the Yahoo Finance API directly without database dependencies.
"""

from datetime import date, datetime
from typing import Optional, Tuple
import pandas as pd
import requests

# HTTP headers for Yahoo Finance API
YAHOO_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
}


def _fetch_yahoo_single(
    ticker: str, start_date: date, end_date: date
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """Fetch one ticker from Yahoo Finance API using direct HTTP requests."""
    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.max.time())

    start_ts = int(start_dt.timestamp())
    end_ts = int(end_dt.timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}?period1={start_ts}&period2={end_ts}&interval=1d"

    try:
        response = requests.get(url, headers=YAHOO_HEADERS, timeout=30)
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

                dates = pd.to_datetime(timestamps, unit="s").normalize()
                adj_close_series = pd.Series(adjclose, index=dates, name=ticker)
                volume_series = pd.Series(vol, index=dates, name=ticker)

                if adj_close_series.index.tz is not None:
                    adj_close_series.index = adj_close_series.index.tz_localize(None)
                if volume_series.index.tz is not None:
                    volume_series.index = volume_series.index.tz_localize(None)

                adj_close_series = adj_close_series[~adj_close_series.index.duplicated(keep="first")]
                volume_series = volume_series[~volume_series.index.duplicated(keep="first")]

                return adj_close_series, volume_series

        print(f"No valid data returned for {ticker}")
        return None, None

    except Exception as e:
        print(f"Failed to fetch {ticker}: {e}")
        return None, None


def test_fetch_single_ticker():
    """Test fetching a single ticker from Yahoo Finance."""
    print("=" * 50)
    print("Testing Yahoo Finance Data Fetcher")
    print("=" * 50)

    # Test with AAPL
    ticker = "AAPL"
    start_date = date(2024, 1, 1)
    end_date = date(2024, 12, 31)

    print(f"\nFetching {ticker} from {start_date} to {end_date}...")

    adj_close, volume = _fetch_yahoo_single(ticker, start_date, end_date)

    if adj_close is not None:
        print(f"\n✅ Success! Got {len(adj_close)} data points")
        print(f"\nFirst 5 rows of adj_close:")
        print(adj_close.head())
        print(f"\nLast 5 rows of adj_close:")
        print(adj_close.tail())
        print(f"\nDate range: {adj_close.index.min()} to {adj_close.index.max()}")
        print(f"Price range: ${adj_close.min():.2f} - ${adj_close.max():.2f}")
    else:
        print("\n❌ Failed to fetch data")
        return False

    # Test with multiple tickers
    print("\n" + "=" * 50)
    print("Testing multiple tickers...")
    print("=" * 50)

    test_tickers = ["MSFT", "NVDA", "^VIX"]
    for t in test_tickers:
        adj, vol = _fetch_yahoo_single(t, start_date, end_date)
        if adj is not None:
            print(f"✅ {t}: {len(adj)} rows, latest price: {adj.iloc[-1]:.2f}")
        else:
            print(f"❌ {t}: fetch failed")

    print("\n" + "=" * 50)
    print("Test completed!")
    print("=" * 50)
    return True


if __name__ == "__main__":
    test_fetch_single_ticker()
