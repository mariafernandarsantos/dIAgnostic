"""
Configuração do mecanismo do banco de dados e metadados.
"""
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import DATABASE_URL

DATABASE_URL = "sqlite:///./db.sqlite3"

# Engine (apenas se for usar com sessões)
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}
)

# Metadados para registrar tabelas
metadata = MetaData()

Base = declarative_base()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
