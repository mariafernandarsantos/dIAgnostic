from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.prediction import ChatRequest, ChatResponse, ChatHistoryResponse
from app.services.ai import chat_with_ai
from app.utils.authUtils import get_current_user
from app.models.user import User
from app.crud.chat import save_chat_message, get_chat_history, create_chat_session, get_user_chat_sessions
from app.crud.prediction import create_prediction_record, get_user_predictions
from app.db.database import get_db
from typing import List

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("/", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest, 
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Conversa com o assistente de IA e realiza predições médicas no chat."""
    try:
        # Criar sessão se não existir
        session_id = request.session_id or await create_chat_session(db, current_user.id)
        
        # Recuperar histórico de chat da sessão
        chat_history = await get_chat_history(db, current_user.id, session_id, limit=10)
        
        # Recuperar histórico de predições para contexto
        prediction_history = await get_user_predictions(db, current_user.id, limit=5)
        
        # Interagir com o assistente de IA
        response_text = await chat_with_ai(
            message=request.message,
            chat_history=chat_history,
            prediction_history=prediction_history,
            user_context=request.context
        )
        
        # Salvar a conversa no histórico
        await save_chat_message(
            db=db,
            user_id=current_user.id,
            session_id=session_id,
            message=request.message,
            response=response_text,
            message_type="chat"
        )
        
        return {
            "response": response_text,
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar chat: {str(e)}")

@router.get("/history", response_model=List[ChatHistoryResponse])
async def get_chat_history_endpoint(
    session_id: str = None,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Recupera o histórico de chat do usuário."""
    try:
        history = await get_chat_history(db, current_user.id, session_id)
        return [
            {
                "id": chat.id,
                "session_id": chat.session_id,
                "message": chat.message,
                "response": chat.response,
                "message_type": chat.message_type,
                "timestamp": chat.timestamp.isoformat()
            }
            for chat in history
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao recuperar histórico: {str(e)}")

@router.get("/sessions")
async def get_chat_sessions_endpoint(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Recupera todas as sessões de chat do usuário."""
    try:
        sessions = await get_user_chat_sessions(db, current_user.id)
        return {"sessions": sessions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao recuperar sessões: {str(e)}")