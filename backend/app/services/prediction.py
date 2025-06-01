"""
Serviços de predição para diagnósticos médicos.
"""
import os
import pickle
import numpy as np
import tensorflow as tf
from typing import Tuple, Dict

# Variáveis globais para os modelos
pneumonia_interpreter = None
pneumonia_input_details = None
pneumonia_output_details = None

diabetes_model = None
diabetes_scaler = None

def load_pneumonia_model():
    global pneumonia_interpreter, pneumonia_input_details, pneumonia_output_details
    if pneumonia_interpreter is None:
        pneumonia_interpreter = tf.lite.Interpreter(model_path="ml_models/pneumonia_model.tflite")
        pneumonia_interpreter.allocate_tensors()
        pneumonia_input_details = pneumonia_interpreter.get_input_details()
        pneumonia_output_details = pneumonia_interpreter.get_output_details()

def load_diabetes_model():
    global diabetes_model, diabetes_scaler
    if diabetes_model is None or diabetes_scaler is None:
        with open("ml_models/diabetes_model.sav", "rb") as model_file:
            diabetes_model = pickle.load(model_file)
        with open("ml_models/diabetes_scaler.sav", "rb") as scaler_file:
            diabetes_scaler = pickle.load(scaler_file)

# def load_ml_models(models_folder: str = "ml_models") -> bool:
#     """
#     Carrega todos os modelos de machine learning necessários.
#     """
#     global pneumonia_interpreter, pneumonia_input_details, pneumonia_output_details
#     global diabetes_model, diabetes_scaler

#     try:
#         # Carregar modelo de pneumonia (TFLite)
#         pneumonia_interpreter = tf.lite.Interpreter(model_path=f"{models_folder}/pneumonia_model.tflite")
#         pneumonia_interpreter.allocate_tensors()
#         pneumonia_input_details = pneumonia_interpreter.get_input_details()
#         pneumonia_output_details = pneumonia_interpreter.get_output_details()

#         # Carregar modelo de diabetes e scaler
#         with open(f"{models_folder}/diabetes_model.sav", "rb") as model_file:
#             diabetes_model = pickle.load(model_file)
#         with open(f"{models_folder}/diabetes_scaler.sav", "rb") as scaler_file:
#             diabetes_scaler = pickle.load(scaler_file)

#         return True
#     except Exception as e:
#         print(f"Erro ao carregar modelos: {e}")
#         return False

def predict_pneumonia(processed_img: np.ndarray) -> Tuple[str, float, float]:
    """
    Faz predição de pneumonia a partir de uma imagem pré-processada.

    A imagem deve estar com shape (1, altura, largura, canais) e dtype float32.
    """
    load_pneumonia_model()
    if pneumonia_interpreter is None:
        raise ValueError("Modelo de pneumonia não carregado")

    processed_img = processed_img.astype(np.float32)

    # Enviar imagem para o modelo
    pneumonia_interpreter.set_tensor(pneumonia_input_details[0]['index'], processed_img)

    # Executar inferência
    pneumonia_interpreter.invoke()

    # Obter saída
    output = pneumonia_interpreter.get_tensor(pneumonia_output_details[0]['index'])

    prediction_value = float(output[0][0])
    result = "PNEUMONIA" if prediction_value > 0.5 else "NORMAL"
    confidence = prediction_value if prediction_value > 0.5 else 1 - prediction_value

    return result, confidence, prediction_value

def predict_diabetes(input_values: list) -> Tuple[str, float]:
    """
    Faz predição de diabetes a partir dos valores de entrada.
    """
    load_diabetes_model()
    if diabetes_model is None or diabetes_scaler is None:
        raise ValueError("Modelo de diabetes ou scaler não carregados")

    scaled_data = diabetes_scaler.transform(np.array(input_values).reshape(1, -1))

    prediction = diabetes_model.predict(scaled_data)
    probability = diabetes_model.predict_proba(scaled_data)[0][1]

    result = "POSITIVE" if prediction[0] == 1 else "NEGATIVE"
    return result, float(probability)

def get_models_status() -> Dict[str, bool]:
    """
    Verifica o status de carregamento dos modelos.
    """
    from app.services.ai import gemini_model

    return {
        "pneumonia": pneumonia_interpreter is not None,
        "diabetes": diabetes_model is not None and diabetes_scaler is not None,
        "gemini": gemini_model is not None
    }