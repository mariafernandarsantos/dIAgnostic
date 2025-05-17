"""
Esquemas de validação para usuários.
"""
from pydantic import BaseModel, EmailStr

class UserCreate(BaseModel):
    """Esquema para criação de usuário."""
    name: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    """Esquema para login de usuário."""
    email: EmailStr
    password: str

class TokenResponse(BaseModel):
    """Esquema para resposta de token após login."""
    access_token: str
    token_type: str = "bearer"
    nome: str

class User(BaseModel):
    """Esquema para representação de usuário."""
    id: str
    name: str
    email: EmailStr
    created_at: str = None

    class Config:
        orm_mode = True