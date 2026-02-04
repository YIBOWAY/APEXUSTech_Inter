import uuid
from datetime import date, datetime

from sqlalchemy import Date, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, TIMESTAMP, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class Strategy(Base):
    __tablename__ = "strategies"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    tags: Mapped[list[str]] = mapped_column(ARRAY(String), default=list)
    method: Mapped[str] = mapped_column(String(30), default="simple")
    lookback_months: Mapped[int] = mapped_column(Integer, default=6)
    decile: Mapped[float] = mapped_column(Float, default=0.2)
    tc_bps: Mapped[int] = mapped_column(Integer, default=5)
    winsor_q: Mapped[float] = mapped_column(Float, default=0.01)
    status: Mapped[str] = mapped_column(String(20), default="active")
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow
    )

    backtest_runs: Mapped[list["BacktestRun"]] = relationship(
        back_populates="strategy", cascade="all, delete-orphan"
    )


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    strategy_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("strategies.id", ondelete="CASCADE")
    )
    started_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), default=datetime.utcnow
    )
    finished_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True))
    status: Mapped[str] = mapped_column(String(20), default="pending")

    # Parameter snapshot
    param_method: Mapped[str | None] = mapped_column(String(30))
    param_lookback: Mapped[int | None] = mapped_column(Integer)
    param_start_date: Mapped[date | None] = mapped_column(Date)
    param_end_date: Mapped[date | None] = mapped_column(Date)

    # Result metrics
    total_return: Mapped[float | None] = mapped_column(Float)
    annual_return: Mapped[float | None] = mapped_column(Float)
    annual_volatility: Mapped[float | None] = mapped_column(Float)
    sharpe_ratio: Mapped[float | None] = mapped_column(Float)
    max_drawdown: Mapped[float | None] = mapped_column(Float)
    win_rate: Mapped[float | None] = mapped_column(Float)
    profit_factor: Mapped[float | None] = mapped_column(Float)
    total_months: Mapped[int | None] = mapped_column(Integer)

    # Series data
    monthly_returns: Mapped[dict | None] = mapped_column(JSONB)
    equity_curve: Mapped[dict | None] = mapped_column(JSONB)

    error_message: Mapped[str | None] = mapped_column(Text)

    strategy: Mapped["Strategy"] = relationship(back_populates="backtest_runs")
    trade_signals: Mapped[list["TradeSignal"]] = relationship(
        back_populates="backtest_run", cascade="all, delete-orphan"
    )
