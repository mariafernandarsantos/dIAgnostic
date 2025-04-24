from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

import numpy as np
import cv2
import uvicorn
import pickle

from tensorflow.keras.models import load_model

from database import database, engine, metadata
from auth import router as auth_router

# Variáveis globais dos modelos
pneumonia_model = None
diabetes_model = None
diabetes_scaler = None

class DiabetesInput(BaseModel):
    pregnancies: int
    glucose: int
    blood_pressure: int
    skin_thickness: int
    insulin: int
    bmi: float
    diabetes_pedigree: float
    age: int

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pneumonia_model, diabetes_model, diabetes_scaler

    # Startup
    metadata.create_all(bind=engine)
    await database.connect()

    try:
        # Load pneumonia model
        pneumonia_model = load_model('./pneumonia_detection_model.h5')
        print("Pneumonia model loaded successfully!")
        
        # Load diabetes model and scaler
        with open("diabetes_model.sav", "rb") as model_file:
            diabetes_model = pickle.load(model_file)
        with open("diabetes_scaler.sav", "rb") as scaler_file:
            diabetes_scaler = pickle.load(scaler_file)
        print("Diabetes model and scaler loaded successfully!")
    except Exception as e:
        print(f"Error loading models: {e}")

    yield  # App executa aqui

    # Shutdown
    await database.disconnect()

# Inicialização do app
app = FastAPI(
    title="dIAgnostic API",
    description="API for medical diagnostics including pneumonia detection and diabetes prediction",
    version="1.0.0",
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

        return img
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

@app.post("/predict/pneumonia", response_class=JSONResponse)
async def predict_pneumonia(file: UploadFile = File(...)):
    """
    Predict pneumonia from an uploaded chest X-ray image.
    
    - **file**: Chest X-ray image file
    
    Returns:
        JSON with diagnosis and confidence score
    """
    if pneumonia_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
        
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
        
    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)
        prediction = pneumonia_model.predict(processed_img)
            
        # Interpret result (threshold at 0.5)
        result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
        confidence = float(prediction[0][0]) if prediction[0][0] > 0.5 else float(1 - prediction[0][0])
            
        return {
            "filename": file.filename,
            "diagnosis": result,
            "confidence": confidence,
            "raw_prediction": float(prediction[0][0])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/predict/diabetes", response_class=JSONResponse)
async def predict_diabetes(input_data: DiabetesInput):
    """
    Predict diabetes based on input parameters.
    
    - **input_data**: Various health metrics for diabetes prediction
    
    Returns:
        JSON with diagnosis and probability
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
        
        return {
            "diagnosis": result,
            "probability": float(probability),
            "message": "Diabetes detected" if prediction[0] == 1 else "No diabetes detected"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing diabetes prediction: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "Welcome to the dIAgnostic API",
        "endpoints": {
            "/predict/pneumonia": "POST an X-ray image to get pneumonia prediction",
            "/predict/diabetes": "POST health metrics to get diabetes prediction",
            "/health": "Check if the API is running"
        }
    }

@app.get("/health")
async def health_check():
    return {
        "status": "ok", 
        "models_loaded": {
            "pneumonia": pneumonia_model is not None,
            "diabetes": diabetes_model is not None and diabetes_scaler is not None
        }
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)