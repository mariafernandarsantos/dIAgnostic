"""
Arquivo init para o pacote API que inicializa e configura todas as rotas.
"""
from fastapi import APIRouter
from app.api.routes import auth, predict, chat

# Criar um router principal para a API
api_router = APIRouter()

# Incluir todos os routers das diferentes seções da API
api_router.include_router(auth.router)
api_router.include_router(predict.router)
api_router.include_router(chat.router)