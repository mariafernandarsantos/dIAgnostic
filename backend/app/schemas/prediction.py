from typing import Optional, Dict, Any, List
from pydantic import BaseModel
from datetime import datetime
from pydantic import BaseModel

class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree: float
    age: int
    get_explanation: bool = False

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class PredictionResponse(BaseModel):
    diagnosis: str
    explanation: Optional[str] = None

class PneumoniaPredictionResponse(PredictionResponse):
    filename: str
    confidence: float
    raw_prediction: float
    mensagem: str

class DiabetesPredictionResponse(PredictionResponse):
    probability: float
    message: str

class ChatResponse(BaseModel):
    response: str
    session_id: str

class ChatHistoryResponse(BaseModel):
    id: str
    session_id: str
    message: str
    response: str
    message_type: str
    timestamp: str

class PredictionHistoryResponse(BaseModel):
    id: str
    type: str
    result: str
    confidence: float
    timestamp: str
    additional_notes: Optional[str] = None
    doctor_reviewed: bool
    doctor_notes: Optional[str] = None

class PredictionStatsResponse(BaseModel):
    diabetes: dict
    pneumonia: dict

class PredictionReview(BaseModel):
    notes: str
    confirmed: bool