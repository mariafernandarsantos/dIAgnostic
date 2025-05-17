from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.database import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserLogin, TokenResponse
from app.core.security import get_password_hash, verify_password, create_access_token
from sqlalchemy.future import select

router = APIRouter(prefix="/auth", tags=["authentication"])

@router.post("/register", status_code=201)
async def register(user: UserCreate, db: AsyncSession = Depends(get_db)):
    """
    Registra um novo usuário no sistema.
    
    Args:
        user: Dados do usuário a ser registrado.
        
    Returns:
        Mensagem de sucesso.
        
    Raises:
        HTTPException: Se o email já estiver registrado.
    """
    # Verificar se o email já existe
    result = await db.execute(select(User).where(User.email == user.email))
    existing = result.scalars().first()
    if existing:
        raise HTTPException(status_code=400, detail="Email já registrado")

    
    new_user = User(
        name=user.name,
        email=user.email,
        hashed_password=get_password_hash(user.password)
    )
    
    db.add(new_user)
    await db.commit()
    return {"message": "Usuário registrado com sucesso"}

@router.post("/login", response_model=TokenResponse)
async def login(user: UserLogin, db: AsyncSession = Depends(get_db)):
    """
    Autentica um usuário e retorna um token de acesso.
    
    Args:
        user: Credenciais de login do usuário.
        
    Returns:
        Token de acesso e tipo do token.
        
    Raises:
        HTTPException: Se as credenciais forem inválidas.
    """
    # Buscar usuário pelo email
    result = await db.execute(select(User).where(User.email == user.email))
    db_user = result.scalars().first()
    
    # Verificar senha
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    # Criar token de acesso
    token = create_access_token({"sub": db_user.email})
    return {"access_token": token, "token_type": "bearer"}