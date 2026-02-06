from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/quantplatform"
    SYNC_DATABASE_URL: str = "postgresql://postgres:postgres@localhost:5432/quantplatform"

    YAHOO_HEADERS: dict = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    # XLK ETF 成分股列表 (66只)
    XLK_TICKERS: list[str] = [
    'AAPL', 'ACN', 'ADBE', 'ADI', 'AKAM', 'AMD', 'AMAT', 'ANET', 'APH', 'AVGO',
    'CDNS', 'CDW', 'CRWD', 'CRM', 'CSCO', 'CTSH', 'DDOG', 'DELL', 'ENPH', 'EPAM',
    'FICO', 'FFIV', 'FSLR', 'FTNT', 'GEN', 'GDDY', 'GLW', 'HPE', 'HPQ', 'IBM',
    'INTC', 'INTU', 'IT', 'JBL', 'KEYS', 'KLAC', 'LRCX', 'MCHP', 'MPWR', 'MSI',
    'MSFT', 'MU', 'NOW', 'NVDA', 'NXPI', 'ON', 'ORCL', 'PANW', 'PLTR', 'PTC',
    'QCOM', 'ROP', 'SMCI', 'SNPS', 'STX', 'SWKS', 'TEL', 'TER', 'TDY', 'TXN',
    'TYL', 'TRMB', 'VRSN', 'WDC', 'WDAY', 'ZBRA'
]

    BENCHMARK_TICKERS: list[str] = ["XLK", "^VIX", "^IRX"]

    class Config:
        env_file = ".env"


settings = Settings()
