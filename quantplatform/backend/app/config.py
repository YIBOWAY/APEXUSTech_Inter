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
        "AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "CSCO", "ACN", "ADBE",
        "IBM", "NOW", "TXN", "QCOM", "INTU", "AMAT", "PANW", "MU", "ADI", "LRCX",
        "KLAC", "SNPS", "CDNS", "APH", "MSI", "CRWD", "NXPI", "MCHP", "TEL", "ROP",
        "FTNT", "ADSK", "DELL", "IT", "GLW", "HPQ", "MPWR", "ON", "FSLR", "ANSS",
        "CDW", "TYL", "KEYS", "ZBRA", "HPE", "TDY", "TRMB", "PTC", "VRSN", "STX",
        "NTAP", "SWKS", "GEN", "TER", "JNPR", "LDOS", "FFIV", "AKAM", "EPAM", "QRVO",
        "WDC", "ENPH", "INTC", "PLTR", "MSTR", "SMCI",
    ]

    BENCHMARK_TICKERS: list[str] = ["XLK", "^VIX", "^IRX"]

    class Config:
        env_file = ".env"


settings = Settings()
