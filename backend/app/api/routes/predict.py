"""
Rotas para serviços de predição médica.
"""
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse

from app.services.prediction import predict_pneumonia, predict_diabetes
from app.schemas.prediction import DiabetesInput, PneumoniaPredictionResponse, DiabetesPredictionResponse
from app.services.ai import get_pneumonia_explanation, get_diabetes_explanation
from app.utils.image import preprocess_image
from app.utils.auth import get_current_user
from app.models.user import User


router = APIRouter(prefix="/predict", tags=["predictions"])

@router.post("/pneumonia", response_model=PneumoniaPredictionResponse, response_class=JSONResponse)
async def predict_pneumonia_endpoint(file: UploadFile = File(...), get_explanation: bool = False, current_user: User = Depends(get_current_user)):
    """
    Prediz pneumonia a partir de uma imagem de raio-X enviada.
    
    Args:
        file: Arquivo de imagem de raio-X
        get_explanation: Definido como true para obter explicação detalhada da IA em português
        
    Returns:
        JSON com diagnóstico, pontuação de confiança e explicação opcional em português
    """
    # Verificar se o arquivo é uma imagem
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
        
        # Se a explicação for solicitada, obtê-la do Gemini
        if get_explanation:
            explanation = await get_pneumonia_explanation(contents, result, confidence)
            response["explanation"] = explanation
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar imagem: {str(e)}")

@router.post("/diabetes", response_model=DiabetesPredictionResponse, response_class=JSONResponse)
async def predict_diabetes_endpoint(input_data: DiabetesInput, current_user: User = Depends(get_current_user)):
    """
    Prediz diabetes com base em parâmetros de entrada.
    
    Args:
        input_data: Várias métricas de saúde para predição de diabetes
        
    Returns:
        JSON com diagnóstico, probabilidade e explicação opcional em português
    """
    try:
        # Extrair valores dos dados de entrada
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
        
        # Obter predição
        result, probability = predict_diabetes(input_values)
        
        result_pt = "POSITIVO" if result == "POSITIVE" else "NEGATIVO"
        
        response = {
            "diagnosis": result_pt,
            "probability": probability,
            "message": "Diabetes detectado" if result == "POSITIVE" else "Diabetes não detectado"
        }
        
        # Se a explicação for solicitada, obtê-la do Gemini
        if input_data.get_explanation:
            explanation = await get_diabetes_explanation(input_data, result, probability)
            response["explanation"] = explanation
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar predição de diabetes: {str(e)}")