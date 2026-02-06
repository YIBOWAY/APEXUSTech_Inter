from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from app.config import settings

engine = create_async_engine(settings.DATABASE_URL, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        yield session


def create_background_session():
    """Create an isolated engine + session for use in background threads.

    Each background thread needs its own engine because asyncpg connections
    are bound to a specific event loop.
    """
    bg_engine = create_async_engine(settings.DATABASE_URL, echo=False)
    bg_session = async_sessionmaker(bg_engine, class_=AsyncSession, expire_on_commit=False)
    return bg_engine, bg_session
