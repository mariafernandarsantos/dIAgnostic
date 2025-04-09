from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import uvicorn

app = FastAPI(
    title="Pneumonia Detection API",
    description="API for detecting pneumonia in chest X-ray images",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
async def load_ml_model():
    """Load the ML model on startup."""
    global model
    try:
        model = load_model('./pneumonia_detection_model.h5')
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")

def preprocess_image(image_bytes):
    """Preprocess the image from bytes for prediction."""
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        # Decode the image
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Could not decode image")
            
        # Resize to match the model's expected input
        img = cv2.resize(img, (150, 150))
        
        # Normalize pixel values
        img = img / 255.0
        
        # Reshape for the model (add batch and channel dimensions)
        img = np.reshape(img, (-1, 150, 150, 1))
        
        return img
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

@app.post("/predict", response_class=JSONResponse)
async def predict_pneumonia(file: UploadFile = File(...)):
    """
    Predict pneumonia from an uploaded chest X-ray image.
    
    - **file**: Chest X-ray image file
    
    Returns:
        JSON with diagnosis and confidence score
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Check if the file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        contents = await file.read()
        processed_img = preprocess_image(contents)
        prediction = model.predict(processed_img)
        
        # Interpret result (threshold at 0.5)
        # Note: Based on the previous conversation, there might be confusion
        # about the label mapping. Adjust this logic if needed.
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

@app.get("/")
async def root():
    return {"message": "Welcome to the Pneumonia Detection API", 
            "endpoints": {
                "/predict": "POST an image to get pneumonia prediction",
                "/health": "Check if the API is running"
            }}

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)