"""
Serviços de predição para diagnósticos médicos.
"""
import os
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from typing import Tuple, Dict

# Variáveis globais para os modelos
pneumonia_model = None
diabetes_model = None
diabetes_scaler = None

def load_ml_models(models_folder: str = "ml_models") -> bool:
    """
    Carrega todos os modelos de machine learning necessários.
    
    Args:
        models_folder: Diretório onde os modelos estão armazenados.
        
    Returns:
        True se todos os modelos foram carregados com sucesso, False caso contrário.
    """
    global pneumonia_model, diabetes_model, diabetes_scaler
    
    try:
        # Carregar modelo de pneumonia
        pneumonia_model = load_model(f"{models_folder}/pneumonia_detection_model.h5")
        
        # Carregar modelo de diabetes e scaler
        with open(f"{models_folder}/diabetes_model.sav", "rb") as model_file:
            diabetes_model = pickle.load(model_file)
        with open(f"{models_folder}/diabetes_scaler.sav", "rb") as scaler_file:
            diabetes_scaler = pickle.load(scaler_file)
            
        return True
    except Exception as e:
        print(f"Erro ao carregar modelos: {e}")
        return False

def predict_pneumonia(processed_img: np.ndarray) -> Tuple[str, float, float]:
    """
    Faz predição de pneumonia a partir de uma imagem pré-processada.
    
    Args:
        processed_img: Imagem pré-processada como array numpy.
        
    Returns:
        Uma tupla contendo (diagnóstico, confiança, valor bruto da predição).
        
    Raises:
        ValueError: Se o modelo não estiver carregado.
    """
    if pneumonia_model is None:
        raise ValueError("Modelo de pneumonia não carregado")
    
    # Fazer predição
    prediction = pneumonia_model.predict(processed_img)
    
    # Interpretar resultado (limite em 0.5)
    result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
    confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
    
    return result, confidence, float(prediction[0][0])

def predict_diabetes(input_values: list) -> Tuple[str, float]:
    """
    Faz predição de diabetes a partir dos valores de entrada.
    
    Args:
        input_values: Lista de valores para predição (na ordem correta).
        
    Returns:
        Uma tupla contendo (diagnóstico, probabilidade).
        
    Raises:
        ValueError: Se o modelo ou scaler não estiverem carregados.
    """
    if diabetes_model is None or diabetes_scaler is None:
        raise ValueError("Modelo de diabetes ou scaler não carregados")
    
    # Escalar os dados de entrada
    scaled_data = diabetes_scaler.transform(np.array(input_values).reshape(1, -1))
    
    # Fazer predição
    prediction = diabetes_model.predict(scaled_data)
    probability = diabetes_model.predict_proba(scaled_data)[0][1]
    
    result = "POSITIVE" if prediction[0] == 1 else "NEGATIVE"
    
    return result, float(probability)

def get_models_status() -> Dict[str, bool]:
    """
    Verifica o status de carregamento dos modelos.
    
    Returns:
        Dicionário com informações sobre quais modelos estão carregados.
    """
    from app.services.ai import gemini_model
    
    return {
        "pneumonia": pneumonia_model is not None,
        "diabetes": diabetes_model is not None and diabetes_scaler is not None,
        "gemini": gemini_model is not None
    }