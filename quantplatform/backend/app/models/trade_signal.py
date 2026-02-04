from datetime import date

from sqlalchemy import BigInteger, Date, Float, ForeignKey, Index, String, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class TradeSignal(Base):
    __tablename__ = "trade_signals"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    backtest_run_id: Mapped[str] = mapped_column(
        UUID(as_uuid=True), ForeignKey("backtest_runs.id", ondelete="CASCADE")
    )
    signal_date: Mapped[date] = mapped_column(Date, nullable=False)
    ticker: Mapped[str] = mapped_column(String(20), nullable=False)
    direction: Mapped[str] = mapped_column(String(10), nullable=False)  # BUY / SELL
    weight: Mapped[float] = mapped_column(Float, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    price: Mapped[float | None] = mapped_column(Float)
    pnl: Mapped[float | None] = mapped_column(Float)

    __table_args__ = (
        UniqueConstraint("backtest_run_id", "signal_date", "ticker", name="uq_run_date_ticker"),
        Index("idx_ts_run", "backtest_run_id"),
        Index("idx_ts_date", "signal_date"),
    )

    backtest_run = relationship("BacktestRun", back_populates="trade_signals")
