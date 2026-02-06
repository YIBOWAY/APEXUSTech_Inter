"""Seed script to populate initial strategy data."""

import asyncio
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import async_session, engine, Base
from app.models.strategy import Strategy


async def seed_strategies():
    """Create initial XLK Momentum strategies."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as db:
        # Check if strategies already exist
        result = await db.execute(select(Strategy))
        existing = result.scalars().all()
        if existing:
            print(f"Database already has {len(existing)} strategies, skipping seed.")
            return

        # Create default strategies
        strategies = [
            Strategy(
                name="XLK Momentum (Risk-Adjusted 6M)",
                description="Long-short momentum strategy on XLK ETF constituents using risk-adjusted scoring with 6-month lookback. Targets top/bottom 20% of stocks monthly.",
                tags=["momentum", "xlk", "tech", "risk-adjusted"],
                method="risk_adjusted",
                lookback_months=6,
                decile=0.2,
                tc_bps=5,
                winsor_q=0.01,
                status="active",
            ),
            Strategy(
                name="XLK Momentum (Simple 3M)",
                description="Classic price momentum strategy with 3-month lookback. More responsive to short-term trends.",
                tags=["momentum", "xlk", "tech", "short-term"],
                method="simple",
                lookback_months=3,
                decile=0.2,
                tc_bps=5,
                winsor_q=0.01,
                status="active",
            ),
            Strategy(
                name="XLK Momentum (Volume-Weighted 6M)",
                description="Momentum strategy incorporating volume signals with 6-month lookback. Higher weight on stocks with increasing volume.",
                tags=["momentum", "xlk", "tech", "volume"],
                method="volume_weighted",
                lookback_months=6,
                decile=0.2,
                tc_bps=5,
                winsor_q=0.01,
                status="active",
            ),
        ]

        for strategy in strategies:
            db.add(strategy)
            print(f"Created strategy: {strategy.name}")

        await db.commit()
        print(f"\nSeeded {len(strategies)} strategies successfully!")


if __name__ == "__main__":
    asyncio.run(seed_strategies())
