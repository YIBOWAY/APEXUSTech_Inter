"""Strategy CRUD API endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.strategy import Strategy, BacktestRun
from app.schemas.strategy import StrategyOut, StrategyCreate, MetricsOut

router = APIRouter(prefix="/strategies", tags=["strategies"])


def _strategy_to_out(strategy: Strategy, latest_run: BacktestRun | None) -> StrategyOut:
    """Convert ORM Strategy to StrategyOut schema."""
    metrics = None
    if latest_run and latest_run.status == "completed":
        metrics = MetricsOut(
            total_return=latest_run.total_return or 0.0,
            sharpe_ratio=latest_run.sharpe_ratio or 0.0,
            max_drawdown=latest_run.max_drawdown or 0.0,
            annual_volatility=latest_run.annual_volatility or 0.0,
            win_rate=latest_run.win_rate or 0.0,
            profit_factor=latest_run.profit_factor or 0.0,
        )

    return StrategyOut(
        id=str(strategy.id),
        name=strategy.name,
        description=strategy.description or "",
        tags=strategy.tags or [],
        metrics=metrics,
        method=strategy.method,
        lookback_months=strategy.lookback_months,
        status=strategy.status,
        created_at=strategy.created_at.isoformat() if strategy.created_at else "",
        updated_at=strategy.updated_at.isoformat() if strategy.updated_at else "",
    )


@router.get("", response_model=list[StrategyOut])
async def list_strategies(
    search: str | None = None,
    sort_by: str = "created_at",
    order: str = "desc",
    db: AsyncSession = Depends(get_db),
):
    """Get all strategies with their latest metrics."""
    stmt = select(Strategy)
    if search:
        stmt = stmt.where(Strategy.name.ilike(f"%{search}%"))

    result = await db.execute(stmt)
    strategies = result.scalars().all()

    # Get latest backtest run for each strategy
    output = []
    for strategy in strategies:
        run_stmt = (
            select(BacktestRun)
            .where(BacktestRun.strategy_id == strategy.id)
            .where(BacktestRun.status == "completed")
            .order_by(BacktestRun.finished_at.desc())
            .limit(1)
        )
        run_result = await db.execute(run_stmt)
        latest_run = run_result.scalar_one_or_none()
        output.append(_strategy_to_out(strategy, latest_run))

    # Sort
    reverse = order == "desc"
    if sort_by == "sharpe_ratio":
        output.sort(key=lambda x: x.metrics.sharpe_ratio if x.metrics else -999, reverse=reverse)
    elif sort_by == "total_return":
        output.sort(key=lambda x: x.metrics.total_return if x.metrics else -999, reverse=reverse)
    elif sort_by == "name":
        output.sort(key=lambda x: x.name, reverse=reverse)
    else:
        output.sort(key=lambda x: x.created_at, reverse=reverse)

    return output


@router.get("/{strategy_id}", response_model=StrategyOut)
async def get_strategy(strategy_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get single strategy by ID."""
    stmt = select(Strategy).where(Strategy.id == strategy_id)
    result = await db.execute(stmt)
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Get latest run
    run_stmt = (
        select(BacktestRun)
        .where(BacktestRun.strategy_id == strategy.id)
        .where(BacktestRun.status == "completed")
        .order_by(BacktestRun.finished_at.desc())
        .limit(1)
    )
    run_result = await db.execute(run_stmt)
    latest_run = run_result.scalar_one_or_none()

    return _strategy_to_out(strategy, latest_run)


@router.post("", response_model=StrategyOut)
async def create_strategy(data: StrategyCreate, db: AsyncSession = Depends(get_db)):
    """Create a new strategy."""
    strategy = Strategy(
        name=data.name,
        description=data.description,
        tags=data.tags,
        method=data.method,
        lookback_months=data.lookback_months,
        decile=data.decile,
        tc_bps=data.tc_bps,
    )
    db.add(strategy)
    await db.commit()
    await db.refresh(strategy)

    return _strategy_to_out(strategy, None)
