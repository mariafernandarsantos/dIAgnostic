"""
Arquivo principal da aplicação FastAPI.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import (
    API_TITLE, 
    API_DESCRIPTION, 
    API_VERSION,
    CORS_ORIGINS,
    CORS_ALLOW_CREDENTIALS,
    CORS_ALLOW_METHODS,
    CORS_ALLOW_HEADERS
)

from app.db.database import engine
from app.api import api_router
from app.services.prediction import load_ml_models, get_models_status
from app.db.base import Base

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação, conectando ao banco de dados e carregando modelos.
    """
    # Startup
    print("Iniciando aplicação...")
    
    # Criar tabelas se não existirem
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Tabelas criadas com sucesso!")
    
    # Carregar modelos de ML
    if load_ml_models():
        print("Modelos de ML carregados com sucesso!")
    else:
        print("Falha ao carregar alguns modelos de ML!")
    
    yield  # App executa aqui
    
    # Shutdown
    print("Desligando aplicação...")
    await engine.dispose()
    print("Banco de dados desconectado.")

# Inicialização do app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan
)

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=CORS_ALLOW_METHODS,
    allow_headers=CORS_ALLOW_HEADERS,
)

# Incluir todas as rotas da API
app.include_router(api_router)

@app.get("/", tags=["status"])
async def root():
    """
    Retorna informações básicas sobre a API e seus endpoints.
    """
    return {
        "message": "Bem-vindo à API dIAgnostic aprimorada com integração Gemini LLM",
        "endpoints": {
            "/predict/pneumonia": "POST de uma imagem de raio-X para obter previsão de pneumonia com explicação opcional da IA",
            "/predict/diabetes": "POST de métricas de saúde para obter previsão de diabetes com explicação opcional da IA",
            "/chat": "Converse com o assistente de IA sobre tópicos médicos",
            "/health": "Verifique se a API está funcionando"
        }
    }

@app.get("/health", tags=["status"])
async def health_check():
    """
    Verifica o status de saúde da API e seus componentes.
    """
    return {
        "status": "ok",
        "models_loaded": get_models_status()
    }