from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.schemas.user import UserCreate, UserLogin, TokenResponse
from app.core.security import create_access_token
from app.crud.user import get_user_by_email, create_user
from app.core.security import verify_password

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", status_code=201)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """Registra um novo usuário no sistema."""
    # Verificar se o email já existe
    existing = await get_user_by_email(db, user.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email já registrado")
    
    # Criar novo usuário
    new_user = await create_user(db, user.name, user.email, user.password)
    return {"message": "Usuário registrado com sucesso", "user_id": new_user.id}

@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    """Autentica um usuário e retorna um token de acesso."""
    # Buscar usuário pelo email
    db_user = await get_user_by_email(db, user.email)
    
    # Verificar senha
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    # Criar token de acesso
    token = create_access_token({"sub": db_user.email, "user_id": db_user.id})
    return {
        "access_token": token, 
        "token_type": "bearer", 
        "nome": db_user.name,
        "user_id": db_user.id
    }