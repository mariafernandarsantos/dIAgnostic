from sqlalchemy import Column, String, ForeignKey, DateTime, JSON, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base

class PredictionHistory(Base):
    __tablename__ = "prediction_history"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    prediction_type = Column(String, nullable=False)  # "diabetes" ou "pneumonia"
    result = Column(String, nullable=False)          # "POSITIVO" ou "NEGATIVO"
    confidence = Column(String, nullable=False)      # Valor numérico da confiança
    input_data = Column(JSON, nullable=True)         # Dados completos de entrada
    additional_notes = Column(Text, nullable=True)   # Notas adicionais/explicação
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    is_doctor_reviewed = Column(Boolean, default=False)  # Foi revisado por médico?
    confirmed_by_doctor = Column(Boolean, default=False)  # Revisão confirmada pelo médico
    doctor_notes = Column(Text, nullable=True)       # Anotações do médico

    user = relationship("User", back_populates="prediction_history")