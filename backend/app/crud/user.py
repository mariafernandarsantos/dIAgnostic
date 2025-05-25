from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models.user import User
from app.core.security import get_password_hash
from typing import Optional
import uuid

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    """Busca um usuário pelo e-mail."""
    result = await db.execute(select(User).where(User.email == email))
    return result.scalars().first()

async def get_user_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
    """Busca um usuário pelo ID."""
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalars().first()

async def create_user(db: AsyncSession, name: str, email: str, password: str) -> User:
    """Cria um novo usuário no banco de dados."""
    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(password)
    
    new_user = User(
        id=user_id,
        name=name,
        email=email,
        hashed_password=hashed_password
    )
    
    db.add(new_user)
    await db.commit()
    await db.refresh(new_user)
    return new_user

async def update_user(db: AsyncSession, user_id: str, **kwargs) -> Optional[User]:
    """Atualiza dados do usuário."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalars().first()
    
    if not user:
        return None
    
    for key, value in kwargs.items():
        if hasattr(user, key):
            setattr(user, key, value)
    
    await db.commit()
    await db.refresh(user)
    return user