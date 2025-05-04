from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
from typing import Optional, Dict, Any
import base64
import os

import numpy as np
import cv2
import uvicorn
import pickle
import google.generativeai as genai
from dotenv import load_dotenv

from tensorflow.keras.models import load_model

from database import database, engine, metadata
from auth import router as auth_router

# Load environment variables
load_dotenv()

# Configure Gemini API
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found in environment variables")
else:
    genai.configure(api_key=GEMINI_API_KEY)

# Variáveis globais dos modelos
pneumonia_model = None
diabetes_model = None
diabetes_scaler = None
gemini_model = None

class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree: float
    age: int
    get_explanation: bool = False  # Optional flag to request detailed explanation

class ChatRequest(BaseModel):
    message: str
    context: Optional[Dict[str, Any]] = None
    history: Optional[list] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pneumonia_model, diabetes_model, diabetes_scaler, gemini_model

    # Startup
    metadata.create_all(bind=engine)
    await database.connect()

    try:
        models_folder = "ml_models"
        # Load pneumonia model
        pneumonia_model = load_model(f"{models_folder}/pneumonia_detection_model.h5")
        print("Pneumonia model loaded successfully!")
        
        # Load diabetes model and scaler
        with open(f"{models_folder}/diabetes_model.sav", "rb") as model_file:
            diabetes_model = pickle.load(model_file)
        with open(f"{models_folder}/diabetes_scaler.sav", "rb") as scaler_file:
            diabetes_scaler = pickle.load(scaler_file)
        print("Diabetes model and scaler loaded successfully!")
        
        # Initialize Gemini model if API key is available
        if GEMINI_API_KEY:
            gemini_model = genai.GenerativeModel('gemini-1.5-pro')
            print("Gemini LLM model initialized successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

    yield  # App executa aqui

    # Shutdown
    await database.disconnect()

# Inicialização do app
app = FastAPI(
    title="dIAgnostic API",
    description="API for medical diagnostics including pneumonia detection and diabetes prediction, enhanced with AI explanations",
    version="1.5.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas de autenticação
app.include_router(auth_router)

# Preprocessamento da imagem
def preprocess_image(image_bytes):
    """Preprocess the image from bytes for prediction."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image as color (3 channels)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image")

        # Resize to 256x256
        img = cv2.resize(img, (256, 256))

        # Normalize pixel values
        img = img / 255.0

        # Reshape for the model (batch size of 1)
        img = np.reshape(img, (1, 256, 256, 3))

        return img, cv2.resize(cv2.imdecode(nparr, cv2.IMREAD_COLOR), (256, 256))
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

async def get_pneumonia_explanation(image_bytes, diagnosis, confidence):
    """Get AI explanation for pneumonia diagnosis in Portuguese."""
    if gemini_model is None:
        return "Explicação detalhada não disponível. API Gemini não configurada."
    
    try:
        # Convert image to base64 for Gemini
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Create prompt for Gemini in Portuguese
        diagnosis_pt = "PNEUMONIA" if diagnosis == "PNEUMONIA" else "NORMAL"
        
        prompt = f"""
        Como especialista em imagens médicas, por favor analise esta radiografia de tórax.
        
        O modelo de aprendizado de máquina determinou que esta imagem é {diagnosis_pt} com {confidence:.2f} de confiança.
        
        Por favor, forneça:
        1. Uma explicação detalhada do que você vê nesta radiografia
        2. Indicadores-chave que sugerem {diagnosis_pt}
        3. Quais regiões específicas da imagem mostram evidências que apoiam este diagnóstico
        4. Informações educativas sobre pneumonia que ajudariam o paciente a entender sua condição
        5. Recomendações gerais para alguém com este diagnóstico
        
        Mantenha sua resposta profissional, mas acessível para pacientes sem formação médica.
        Forneça uma análise abrangente em 3-4 parágrafos.
        
        IMPORTANTE: Responda COMPLETAMENTE em português do Brasil.
        """
        
        # Call Gemini API with the image
        response = await gemini_model.generate_content_async(
            [prompt, {"mime_type": "image/jpeg", "data": base64_image}]
        )
        
        return response.text
    except Exception as e:
        return f"Unable to generate detailed explanation: {str(e)}"

async def get_diabetes_explanation(input_data, diagnosis, probability):
    """Get AI explanation for diabetes diagnosis in Portuguese."""
    if gemini_model is None:
        return "Explicação detalhada não disponível. API Gemini não configurada."
    
    try:
        # Create prompt for Gemini in Portuguese
        diagnosis_pt = "POSITIVO" if diagnosis == "POSITIVE" else "NEGATIVO"
        
        prompt = f"""
        Como especialista médico, por favor analise esta previsão de diabetes.
        
        O modelo de aprendizado de máquina determinou um diagnóstico {diagnosis_pt} para diabetes com {probability:.2f} de probabilidade.
        
        Dados do paciente:
        - Gestações: {input_data.pregnancies}
        - Nível de glicose: {input_data.glucose} mg/dL
        - Pressão arterial: {input_data.blood_pressure} mm Hg
        - Espessura da pele: {input_data.skin_thickness} mm
        - Insulina: {input_data.insulin} μU/mL
        - IMC: {input_data.bmi}
        - Função pedigree de diabetes: {input_data.diabetes_pedigree}
        - Idade: {input_data.age} anos
        
        Por favor, forneça:
        1. Uma explicação detalhada de quais fatores provavelmente contribuíram mais para esta previsão
        2. Como cada medição se compara aos intervalos saudáveis típicos
        3. Informações educativas sobre diabetes que ajudariam o paciente a entender a condição
        4. Recomendações personalizadas com base nestas medições específicas
        
        Mantenha sua resposta profissional, mas acessível para pacientes sem formação médica.
        Forneça uma análise abrangente em 3-4 parágrafos.
        
        IMPORTANTE: Responda COMPLETAMENTE em português do Brasil.
        """
        
        # Call Gemini API
        response = await gemini_model.generate_content_async(prompt)
        
        return response.text
    except Exception as e:
        return f"Unable to generate detailed explanation: {str(e)}"

@app.post("/predict/pneumonia", response_class=JSONResponse)
async def predict_pneumonia(file: UploadFile = File(...), get_explanation: bool = False):
    """
    Predict pneumonia from an uploaded chest X-ray image.
    
    - **file**: Chest X-ray image file
    - **get_explanation**: Set to true to get AI-powered detailed explanation in Portuguese
    
    Returns:
        JSON with diagnosis, confidence score, and optional explanation in Portuguese
    """
    if pneumonia_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        contents = await file.read()
        processed_img, _ = preprocess_image(contents)
        prediction = pneumonia_model.predict(processed_img)
            
        # Interpret result (threshold at 0.5)
        result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
        confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
        
        response = {
            "filename": file.filename,
            "diagnosis": result,
            "confidence": confidence,
            "raw_prediction": float(prediction[0][0]),
            "mensagem": "Pneumonia detectada" if prediction[0][0] > 0.5 else "Raio-X normal"
        }
        
        # If explanation is requested, get it from Gemini
        if get_explanation:
            explanation = await get_pneumonia_explanation(contents, result, confidence)
            response["explanation"] = explanation
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/diabetes", response_class=JSONResponse)
async def predict_diabetes(input_data: DiabetesInput):
    """
    Predict diabetes based on input parameters.
    
    - **input_data**: Various health metrics for diabetes prediction
    - **get_explanation**: Set to true in the request body to get AI-powered detailed explanation in Portuguese
    
    Returns:
        JSON with diagnosis, probability, and optional explanation in Portuguese
    """
    if diabetes_model is None or diabetes_scaler is None:
        raise HTTPException(status_code=500, detail="Diabetes model or scaler not loaded")
    
    try:
        # Extract values from input data
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
        
        # Scale the input data
        scaled_data = diabetes_scaler.transform(np.array(input_values).reshape(1, -1))
        
        # Make prediction
        prediction = diabetes_model.predict(scaled_data)
        probability = diabetes_model.predict_proba(scaled_data)[0][1]
        
        result = "POSITIVE" if prediction[0] == 1 else "NEGATIVE"
        result_pt = "POSITIVO" if prediction[0] == 1 else "NEGATIVO"
        
        response = {
            "diagnosis": result_pt,
            "probability": float(probability),
            "message": "Diabetes detectado" if prediction[0] == 1 else "Diabetes não detectado"
        }
        
        # If explanation is requested, get it from Gemini
        if input_data.get_explanation:
            explanation = await get_diabetes_explanation(input_data, result, probability)
            response["explanation"] = explanation
            
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing diabetes prediction: {str(e)}")

@app.post("/chat", response_class=JSONResponse)
async def chat_with_assistant(request: ChatRequest):
    """
    Chat with the AI assistant about medical topics in Portuguese.
    
    - **message**: User's message/question
    - **context**: Optional context information (diagnosis results, etc.)
    - **history**: Optional conversation history
    
    Returns:
        JSON with the assistant's response in Portuguese
    """
    if gemini_model is None:
        raise HTTPException(status_code=500, detail="Gemini model not loaded")
    
    try:
        # Prepare conversation history if provided
        chat = gemini_model.start_chat(history=[])
        
        if request.history:
            for msg in request.history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    chat.send_message(content)
                else:
                    # Add as model response in history
                    chat.history.append({"role": "model", "parts": [content]})
        
        # Format the prompt with context if available
        prompt = request.message
        if request.context:
            context_str = "\n".join([f"{k}: {v}" for k, v in request.context.items()])
            prompt = f"""
            Pergunta do usuário: {request.message}
            
            Informações de contexto:
            {context_str}
            
            Por favor, forneça uma resposta útil, precisa e educativa à pergunta do usuário.
            Concentre-se em explicar conceitos médicos em termos simples, mantendo a precisão.
            
            IMPORTANTE: Responda COMPLETAMENTE em português do Brasil.
            """
        
        # Get response from Gemini
        response = await chat.send_message_async(prompt)
        
        return {
            "response": response.text,
            "conversation_id": id(chat)  # Use a unique identifier for this conversation
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Bem-vindo à API dIAgnostic aprimorada com integração Gemini LLM",
        "endpoints": {
            "/predict/pneumonia": "POST de uma imagem de raio-X para obter previsão de pneumonia com explicação opcional da IA",
            "/predict/diabetes": "POST de métricas de saúde para obter previsão de diabetes com explicação opcional da IA",
            "/chat": "Converse com o assistente de IA sobre tópicos médicos",
            "/health": "Verifique se a API está funcionando"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "models_loaded": {
            "pneumonia": pneumonia_model is not None,
            "diabetes": diabetes_model is not None and diabetes_scaler is not None,
            "gemini": gemini_model is not None
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)