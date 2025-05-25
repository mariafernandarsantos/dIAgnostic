from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, and_
from app.models.chat_history import ChatHistory
from typing import List, Optional
import uuid

async def create_chat_session(db: AsyncSession, user_id: str) -> str:
    """Cria uma nova sessão de chat."""
    session_id = str(uuid.uuid4())
    return session_id

async def save_chat_message(
    db: AsyncSession,
    user_id: str,
    session_id: str,
    message: str,
    response: str,
    message_type: str = "chat"
) -> ChatHistory:
    """Salva uma mensagem e resposta do chat."""
    chat_id = str(uuid.uuid4())
    
    chat_record = ChatHistory(
        id=chat_id,
        user_id=user_id,
        session_id=session_id,
        message=message,
        response=response,
        message_type=message_type
    )
    
    db.add(chat_record)
    await db.commit()
    await db.refresh(chat_record)
    return chat_record

async def get_chat_history(
    db: AsyncSession,
    user_id: str,
    session_id: Optional[str] = None,
    limit: int = 50
) -> List[ChatHistory]:
    """Recupera o histórico de chat do usuário."""
    query = select(ChatHistory).where(ChatHistory.user_id == user_id)
    
    if session_id:
        query = query.where(ChatHistory.session_id == session_id)
    
    query = query.order_by(desc(ChatHistory.timestamp)).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()

async def get_user_chat_sessions(db: AsyncSession, user_id: str) -> List[str]:
    """Recupera todas as sessões de chat de um usuário."""
    result = await db.execute(
        select(ChatHistory.session_id)
        .where(ChatHistory.user_id == user_id)
        .distinct()
        .order_by(desc(ChatHistory.timestamp))
    )
    return [row[0] for row in result.fetchall()]

async def delete_chat_session(db: AsyncSession, user_id: str, session_id: str) -> bool:
    """Deleta uma sessão de chat completa."""
    result = await db.execute(
        select(ChatHistory).where(
            and_(ChatHistory.user_id == user_id, ChatHistory.session_id == session_id)
        )
    )
    chats = result.scalars().all()
    
    if chats:
        for chat in chats:
            await db.delete(chat)
        await db.commit()
        return True
    return False