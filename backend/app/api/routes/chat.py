"""
Rotas para serviço de chat com IA.
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.schemas.prediction import ChatRequest, ChatResponse
from app.services.ai import chat_with_ai
from app.utils.auth import get_current_user
from app.models.user import User

router = APIRouter(tags=["chat"])

@router.post("/chat", response_model=ChatResponse, response_class=JSONResponse)
async def chat_endpoint(request: ChatRequest, current_user: User = Depends(get_current_user)):
    """
    Conversa com o assistente de IA sobre tópicos médicos em português.
    
    Args:
        message: Mensagem/pergunta do usuário
        context: Informações de contexto opcionais (resultados de diagnóstico, etc.)
        history: Histórico opcional de conversas
        
    Returns:
        JSON com a resposta do assistente em português
    """
    try:
        # Obter resposta do serviço de IA
        response_text, conversation_id = await chat_with_ai(
            message=request.message,
            context=request.context,
            history=request.history
        )
        
        if not conversation_id:
            raise HTTPException(status_code=500, detail="Falha ao iniciar conversa com IA")
        
        return {
            "response": response_text,
            "conversation_id": str(conversation_id)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar chat: {str(e)}")