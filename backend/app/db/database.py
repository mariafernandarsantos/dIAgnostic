from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from app.core.config import DATABASE_URL

DATABASE_URL = DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
# Cria o engine assíncrono
engine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Mostra logs das operações (útil para desenvolvimento)
    connect_args={"check_same_thread": False}  # Necessário para SQLite
)

# Cria a fábrica de sessões assíncronas
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)

# Base para os modelos
Base = declarative_base()

async def get_db():
    """
    Fornece uma sessão de banco de dados assíncrona para cada requisição.
    """
    async with AsyncSessionLocal() as session:
        yield session
