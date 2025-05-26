from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from app.services.prediction import predict_pneumonia, predict_diabetes
from app.schemas.prediction import DiabetesInput, PneumoniaPredictionResponse, DiabetesPredictionResponse, PredictionHistoryResponse, PredictionReview
from app.services.ai import get_pneumonia_explanation, get_diabetes_explanation
from app.utils.image import preprocess_image
from app.utils.authUtils import get_current_user
from app.models.user import User
from app.crud.prediction import create_prediction_record, get_user_predictions, get_prediction_statistics, add_doctor_review
from app.db.database import get_db
import pickle
import logging
import numpy as np

# Variáveis globais para os modelos
diabetes_model = None
diabetes_scaler = None
logger = logging.getLogger(__name__)

def load_diabetes_model(models_folder: str = "ml_models"):
    """Carrega o modelo de diabetes e o scaler"""   
    global diabetes_model, diabetes_scaler
    try:
        with open(f"{models_folder}/diabetes_model.sav", "rb") as model_file:
            diabetes_model = pickle.load(model_file)
        with open(f"{models_folder}/diabetes_scaler.sav", "rb") as scaler_file:
            diabetes_scaler = pickle.load(scaler_file)
    except Exception as e:
        logger.error(f"Erro ao carregar modelo diabetes: {e}")
        raise

router = APIRouter(prefix="/predict", tags=["predictions"])

@router.post("/pneumonia", response_model=PneumoniaPredictionResponse)
async def predict_pneumonia_endpoint(
    file: UploadFile = File(...), 
    get_explanation: bool = False,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Prediz pneumonia a partir de uma imagem de raio-X enviada."""
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="O arquivo deve ser uma imagem")
        
    try:
        # Ler e pré-processar a imagem
        contents = await file.read()
        processed_img, _ = preprocess_image(contents)
        
        # Obter predição
        result, confidence, raw_prediction = predict_pneumonia(processed_img)
        response = {
            "filename": file.filename,
            "diagnosis": result,
            "confidence": confidence,
            "raw_prediction": raw_prediction,
            "mensagem": "Pneumonia detectada" if result == "PNEUMONIA" else "Raio-X normal"
        }
        
        # Se a explicação for solicitada, obtê-la do AI
        if get_explanation:
            explanation = await get_pneumonia_explanation(contents, result, confidence)
            response["explanation"] = explanation
        
        # Salvar no histórico
        await create_prediction_record(
            db=db,
            user_id=current_user.id,
            prediction_type="pneumonia",
            result=result,
            confidence=confidence,
            additional_notes=response.get("explanation"),
            input_data={"filename": file.filename}
        )
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")

@router.post("/diabetes", response_model=DiabetesPredictionResponse)
async def predict_diabetes_endpoint(
    input_data: DiabetesInput, 
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        if diabetes_model is None:
            load_diabetes_model()

        result = await predict_diabetes_and_save(
            db=db,
            user_id=current_user.id,
            input_data=input_data
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Erro ao processar predição de diabetes: {str(e)}"
        )

async def predict_diabetes_and_save(
    db: AsyncSession,
    user_id: str,
    input_data: DiabetesInput  # Recebe o objeto Pydantic diretamente
):
    """
    Processa a predição de diabetes e salva no histórico.
    
    Args:
        db: Sessão do banco de dados assíncrono
        user_id: ID do usuário que solicitou a predição
        input_data: Objeto DiabetesInput com os dados de entrada
        
    Returns:
        Dicionário com resultado da predição e metadados
    """
    # Verificar se os modelos estão carregados
    if diabetes_model is None or diabetes_scaler is None:
        raise ValueError("Modelo de diabetes não carregado")

    try:
        # Extrair valores dos dados de entrada (acessa os atributos do objeto Pydantic)
        input_values = [
            input_data.pregnancies,
            input_data.glucose,
            input_data.blood_pressure,
            input_data.skin_thickness,
            input_data.insulin,
            input_data.bmi,
            input_data.diabetes_pedigree,
            input_data.age
        ]

        # Pré-processamento dos dados
        scaled_data = diabetes_scaler.transform(np.array(input_values).reshape(1, -1))

        # Fazer predição
        prediction = diabetes_model.predict(scaled_data)
        probability = diabetes_model.predict_proba(scaled_data)[0][1]

        # Interpretar resultado
        result = "POSITIVE" if prediction[0] == 1 else "NEGATIVE"
        result_pt = "POSITIVO" if result == "POSITIVE" else "NEGATIVO"

        # Converter input_data para dicionário (agora dentro da função)
        input_dict = input_data.dict(exclude={"password", "hashed_password"}, by_alias=True)
        explanation = await get_diabetes_explanation(input_data, result, probability) if input_data.get_explanation else None

        # Salvar no histórico
        await create_prediction_record(
            db=db,
            user_id=user_id,
            prediction_type="diabetes",
            result=result_pt,   
            confidence=probability,
            additional_notes=explanation,
            input_data=input_dict
        )

        return {
            "diagnosis": result_pt,
            "probability": float(probability),
            "input_parameters": input_dict,
            "message": "Diabetes detectado" if result == "POSITIVE" else "Diabetes não detectado",
            "explanation": explanation,
        }

    except Exception as e:
        logger.error(f"Erro na predição de diabetes: {str(e)}")
        raise Exception(f"Falha ao processar predição: {str(e)}")

@router.get("/history", response_model=List[PredictionHistoryResponse])
async def get_prediction_history(   
    prediction_type: Optional[str] = None,
    limit: int = 20,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Recupera o histórico completo de predições do usuário"""
    predictions = await get_user_predictions(
        db, 
        current_user.id,
        prediction_type, 
        limit
    )
    
    return [
        {
            "id": pred.id,
            "type": pred.prediction_type,
            "result": pred.result,
            "confidence": float(pred.confidence),
            "timestamp": pred.timestamp.isoformat(),
            "additional_notes": pred.additional_notes,
            "doctor_reviewed": pred.is_doctor_reviewed,
            "confirmed_by_doctor": pred.confirmed_by_doctor,
            "doctor_notes": pred.doctor_notes
        }
        for pred in predictions
    ]

@router.get("/stats")
async def get_prediction_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Retorna estatísticas das predições do usuário"""
    return await get_prediction_statistics(db, current_user.id)

@router.post("/{prediction_id}/review")
async def review_prediction(
    prediction_id: str,
    review: PredictionReview,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Permite que um médico revise uma predição"""
    return await add_doctor_review(
        db,
        prediction_id,
        current_user.id,
        review.notes,
        review.confirmed
    )
