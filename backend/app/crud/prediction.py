from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import desc, and_, func, cast, Float, case
from fastapi.exceptions import HTTPException
from app.models.prediction_history import PredictionHistory
from typing import List, Optional
import uuid

async def create_prediction_record(
    db: AsyncSession,
    user_id: str,
    prediction_type: str,
    result: str,
    confidence: float,
    input_data: Optional[dict] = None,
    additional_notes: Optional[str] = None,
) -> PredictionHistory:
    """Cria um novo registro de predição."""
    prediction = PredictionHistory(
        id=str(uuid.uuid4()),
        user_id=user_id,
        prediction_type=prediction_type,
        result=result,
        confidence=str(confidence),
        input_data=input_data,
        additional_notes=additional_notes,
        is_doctor_reviewed=False,
    )
    
    db.add(prediction)
    await db.commit()
    await db.refresh(prediction)
    return prediction

async def get_user_predictions(
    db: AsyncSession,
    user_id: str,
    prediction_type: Optional[str] = None,
    limit: int = 50
) -> List[PredictionHistory]:
    """Recupera o histórico de predições de um usuário."""
    query = select(PredictionHistory).where(PredictionHistory.user_id == user_id)
    
    if prediction_type:
        query = query.where(PredictionHistory.prediction_type == prediction_type)
    
    query = query.order_by(desc(PredictionHistory.timestamp)).limit(limit)
    result = await db.execute(query)
    return result.scalars().all()

async def get_prediction_by_id(db: AsyncSession, prediction_id: str) -> Optional[PredictionHistory]:
    """Busca uma predição específica pelo ID."""
    result = await db.execute(select(PredictionHistory).where(PredictionHistory.id == prediction_id))
    return result.scalars().first()

async def delete_prediction(db: AsyncSession, prediction_id: str, user_id: str) -> bool:
    """Deleta uma predição específica do usuário."""
    result = await db.execute(
        select(PredictionHistory).where(
            and_(PredictionHistory.id == prediction_id, PredictionHistory.user_id == user_id)
        )
    )
    prediction = result.scalars().first()
    
    if prediction:
        await db.delete(prediction)
        await db.commit()
        return True
    return False

async def get_prediction_statistics(
    db: AsyncSession, 
    user_id: str
) -> dict:
    """Retorna estatísticas das predições do usuário"""
    result = await db.execute(
        select(
            PredictionHistory.prediction_type,
            func.count().label("total"),
            func.avg(cast(PredictionHistory.confidence, Float)).label("avg_confidence"),
            func.sum(case((PredictionHistory.result == "POSITIVO", 1), else_=0)).label("positive_count")
        )
        .where(PredictionHistory.user_id == user_id)
        .group_by(PredictionHistory.prediction_type)
    )
    
    stats = {}
    for row in result:
        stats[row.prediction_type] = {
            "total": row.total,
            "avg_confidence": float(row.avg_confidence) if row.avg_confidence else 0,
            "positive_rate": row.positive_count / row.total if row.total else 0
        }
    
    return stats

async def add_doctor_review(
    db: AsyncSession,
    prediction_id: str,
    user_id: str,
    notes: str,
    confirmed: bool
) -> PredictionHistory:
    """Adiciona revisão médica a uma predição"""
    prediction = await get_prediction_by_id(db, prediction_id)
    
    if not prediction or prediction.user_id != user_id:
        raise HTTPException(status_code=404, detail="Predição não encontrada")
    
    prediction.is_doctor_reviewed = True
    prediction.confirmed_by_doctor = confirmed
    prediction.doctor_notes = notes
    
    await db.commit()
    await db.refresh(prediction)
    return prediction