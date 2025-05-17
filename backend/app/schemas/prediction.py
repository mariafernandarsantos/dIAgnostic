"""
Esquemas de validação para predições e chat.
"""
from typing import Optional, Dict, Any, List
from pydantic import BaseModel

class DiabetesInput(BaseModel):
    """Esquema para entrada de predição de diabetes."""
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree: float
    age: int
    get_explanation: bool = False  # Flag opcional para solicitar explicação detalhada

class ChatRequest(BaseModel):
    """Esquema para solicitação de chat."""
    message: str
    context: Optional[Dict[str, Any]] = None
    history: Optional[List[Dict[str, Any]]] = None

class PredictionResponse(BaseModel):
    """Esquema base para respostas de predição."""
    diagnosis: str
    explanation: Optional[str] = None

class PneumoniaPredictionResponse(PredictionResponse):
    """Esquema para resposta de predição de pneumonia."""
    filename: str
    confidence: float
    raw_prediction: float
    mensagem: str

class DiabetesPredictionResponse(PredictionResponse):
    """Esquema para resposta de predição de diabetes."""
    probability: float
    message: str

class ChatResponse(BaseModel):
    """Esquema para resposta de chat."""
    response: str
    conversation_id: str