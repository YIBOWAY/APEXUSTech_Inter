"""Backtest execution and result retrieval API endpoints.

Uses background threads for long-running backtests to avoid HTTP timeout.
"""

import asyncio
import logging
import threading
from datetime import date, datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db, create_background_session
from app.models.strategy import Strategy, BacktestRun
from app.models.trade_signal import TradeSignal
from app.schemas.backtest import (
    BacktestRequest,
    BacktestRunOut,
    EquityPointOut,
    TradeOut,
)
from app.services.data_fetcher import fetch_and_cache_prices
from app.services.momentum_strategy import XLK_TICKERS, run_full_backtest

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtest", tags=["backtest"])


async def _run_backtest_async(
    run_id: UUID, strategy_params: dict, request: BacktestRequest
):
    """Execute backtest in background with its own engine + session.

    strategy_params is a plain dict (not ORM object) to avoid cross-session issues.
    """
    bg_engine, bg_session_factory = create_background_session()
    try:
        async with bg_session_factory() as db:
            try:
                stmt = select(BacktestRun).where(BacktestRun.id == run_id)
                result = await db.execute(stmt)
                backtest_run = result.scalar_one()

                # Fetch price data
                logger.info(f"[BG] Fetching data for {len(XLK_TICKERS)} tickers...")
                tickers_with_benchmark = XLK_TICKERS + ["XLK", "^VIX", "^IRX"]

                adj_close, volume = await fetch_and_cache_prices(
                    db, tickers_with_benchmark, request.start_date, request.end_date
                )

                if adj_close.empty:
                    raise ValueError("No price data available")

                # Run backtest
                logger.info(f"[BG] Running backtest: method={backtest_run.param_method}, lookback={backtest_run.param_lookback}")
                results = run_full_backtest(
                    adj_close=adj_close,
                    volume=volume,
                    benchmark_ticker="XLK",
                    lookback_months=backtest_run.param_lookback,
                    method=backtest_run.param_method,
                    decile=strategy_params["decile"],
                    tc_bps=strategy_params["tc_bps"],
                    winsor_q=strategy_params["winsor_q"],
                )

                # Update backtest run with results
                metrics = results["metrics"]
                backtest_run.status = "completed"
                backtest_run.finished_at = datetime.now(timezone.utc)
                backtest_run.total_return = metrics.get("total_return")
                backtest_run.annual_return = metrics.get("annual_return")
                backtest_run.annual_volatility = metrics.get("annual_volatility")
                backtest_run.sharpe_ratio = metrics.get("sharpe_ratio")
                backtest_run.max_drawdown = metrics.get("max_drawdown")
                backtest_run.win_rate = metrics.get("win_rate")
                backtest_run.profit_factor = metrics.get("profit_factor")
                backtest_run.total_months = metrics.get("total_months")
                backtest_run.monthly_returns = results["monthly_returns"]
                backtest_run.equity_curve = results["equity_curve"]

                # Save trade signals in bulk
                trade_signal_objects = [
                    TradeSignal(
                        backtest_run_id=backtest_run.id,
                        signal_date=datetime.strptime(sig["signal_date"], "%Y-%m-%d").date(),
                        ticker=sig["ticker"],
                        direction=sig["direction"],
                        weight=sig["weight"],
                        score=sig.get("score"),
                        price=sig.get("price"),
                    )
                    for sig in results["trade_signals"]
                ]
                db.add_all(trade_signal_objects)

                await db.commit()
                logger.info(f"[BG] Backtest completed: sharpe={backtest_run.sharpe_ratio:.3f}, total_return={backtest_run.total_return:.2%}")

            except Exception as e:
                logger.exception(f"[BG] Backtest failed: {e}")
                await db.rollback()
                stmt = select(BacktestRun).where(BacktestRun.id == run_id)
                result = await db.execute(stmt)
                backtest_run = result.scalar_one()
                backtest_run.status = "failed"
                backtest_run.error_message = str(e)
                backtest_run.finished_at = datetime.now(timezone.utc)
                await db.commit()
    finally:
        await bg_engine.dispose()


def _start_background_backtest(run_id: UUID, strategy_params: dict, request: BacktestRequest):
    """Start backtest in a new thread with its own event loop and DB engine."""
    def run_in_thread():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(
                _run_backtest_async(run_id, strategy_params, request)
            )
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()


@router.post("/{strategy_id}/run", response_model=BacktestRunOut)
async def run_backtest(
    strategy_id: UUID,
    request: BacktestRequest,
    db: AsyncSession = Depends(get_db),
):
    """Start backtest asynchronously and return run ID immediately."""
    # Get strategy
    stmt = select(Strategy).where(Strategy.id == strategy_id)
    result = await db.execute(stmt)
    strategy = result.scalar_one_or_none()

    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")

    # Create backtest run record
    backtest_run = BacktestRun(
        strategy_id=strategy_id,
        status="running",
        param_method=request.method or strategy.method,
        param_lookback=request.lookback_months or strategy.lookback_months,
        param_start_date=request.start_date,
        param_end_date=request.end_date,
    )
    db.add(backtest_run)
    await db.commit()
    await db.refresh(backtest_run)

    # Extract strategy params as plain dict (ORM objects can't cross sessions)
    strategy_params = {
        "decile": strategy.decile,
        "tc_bps": strategy.tc_bps,
        "winsor_q": strategy.winsor_q,
    }

    # Start background task - return immediately
    _start_background_backtest(backtest_run.id, strategy_params, request)

    logger.info(f"Backtest started in background: run_id={backtest_run.id}")
    return _run_to_out(backtest_run)


@router.get("/runs/{run_id}/status", response_model=BacktestRunOut)
async def get_backtest_status(run_id: UUID, db: AsyncSession = Depends(get_db)):
    """Poll backtest run status."""
    stmt = select(BacktestRun).where(BacktestRun.id == run_id)
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="Backtest run not found")

    return _run_to_out(run)


@router.get("/{strategy_id}/latest", response_model=BacktestRunOut)
async def get_latest_backtest(strategy_id: UUID, db: AsyncSession = Depends(get_db)):
    """Get latest completed backtest run for a strategy."""
    stmt = (
        select(BacktestRun)
        .where(BacktestRun.strategy_id == strategy_id)
        .where(BacktestRun.status == "completed")
        .order_by(BacktestRun.finished_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        raise HTTPException(status_code=404, detail="No completed backtest found")

    return _run_to_out(run)


@router.get("/{strategy_id}/equity", response_model=list[EquityPointOut])
async def get_equity_curve(
    strategy_id: UUID,
    run_id: UUID | None = None,
    db: AsyncSession = Depends(get_db),
):
    """Get equity curve for a backtest run."""
    if run_id:
        stmt = select(BacktestRun).where(BacktestRun.id == run_id)
    else:
        stmt = (
            select(BacktestRun)
            .where(BacktestRun.strategy_id == strategy_id)
            .where(BacktestRun.status == "completed")
            .order_by(BacktestRun.finished_at.desc())
            .limit(1)
        )

    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run or not run.equity_curve:
        return []

    return [EquityPointOut(**pt) for pt in run.equity_curve]


@router.get("/{strategy_id}/monthly-returns")
async def get_monthly_returns(
    strategy_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get monthly returns from the latest completed backtest."""
    stmt = (
        select(BacktestRun)
        .where(BacktestRun.strategy_id == strategy_id)
        .where(BacktestRun.status == "completed")
        .order_by(BacktestRun.finished_at.desc())
        .limit(1)
    )
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run or not run.monthly_returns:
        return []

    return run.monthly_returns


@router.get("/{strategy_id}/trades", response_model=list[TradeOut])
async def get_trades(
    strategy_id: UUID,
    run_id: UUID | None = None,
    page: int = 1,
    size: int = 50,
    db: AsyncSession = Depends(get_db),
):
    """Get trade signals for a backtest run."""
    if run_id:
        stmt = select(BacktestRun).where(BacktestRun.id == run_id)
    else:
        stmt = (
            select(BacktestRun)
            .where(BacktestRun.strategy_id == strategy_id)
            .where(BacktestRun.status == "completed")
            .order_by(BacktestRun.finished_at.desc())
            .limit(1)
        )
    result = await db.execute(stmt)
    run = result.scalar_one_or_none()

    if not run:
        return []

    offset = (page - 1) * size
    stmt = (
        select(TradeSignal)
        .where(TradeSignal.backtest_run_id == run.id)
        .order_by(TradeSignal.signal_date.desc())
        .offset(offset)
        .limit(size)
    )
    result = await db.execute(stmt)
    signals = result.scalars().all()

    return [
        TradeOut(
            id=str(sig.id),
            symbol=sig.ticker,
            direction=sig.direction,
            price=sig.price or 0.0,
            quantity=sig.weight,
            timestamp=sig.signal_date.isoformat() if sig.signal_date else "",
            commission=0.0,
            pnl=sig.pnl,
            status="executed",
        )
        for sig in signals
    ]


def _run_to_out(run: BacktestRun) -> BacktestRunOut:
    """Convert ORM BacktestRun to BacktestRunOut schema."""
    return BacktestRunOut(
        id=str(run.id),
        strategy_id=str(run.strategy_id),
        status=run.status,
        started_at=run.started_at.isoformat() if run.started_at else "",
        finished_at=run.finished_at.isoformat() if run.finished_at else None,
        param_method=run.param_method,
        param_lookback=run.param_lookback,
        param_start_date=run.param_start_date.isoformat() if run.param_start_date else None,
        param_end_date=run.param_end_date.isoformat() if run.param_end_date else None,
        total_return=run.total_return,
        annual_return=run.annual_return,
        annual_volatility=run.annual_volatility,
        sharpe_ratio=run.sharpe_ratio,
        max_drawdown=run.max_drawdown,
        win_rate=run.win_rate,
        profit_factor=run.profit_factor,
        total_months=run.total_months,
        error_message=run.error_message,
    )
