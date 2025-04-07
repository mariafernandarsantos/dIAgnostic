from tensorflow.keras.models import load_model
import numpy as np
import cv2

model = load_model('./models/saved_models/pneumonia_detection_model.h5')

def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (150, 150))
    
    # Normalize pixel values
    img = img / 255.0
    
    # Reshape for the model (add batch and channel dimensions)
    img = np.reshape(img, (1, 150, 150, 1))
    
    return img

def predict_pneumonia(image_path):
    processed_img = preprocess_image(image_path)
    prediction = model.predict(processed_img)
    
    # Get result (threshold at 0.5)
    result = "PNEUMONIA" if prediction[0][0] > 0.5 else "NORMAL"
    confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
    
    return result, confidence

result, confidence = predict_pneumonia('./ml_pipeline/dataset/IM-0003-0001.jpeg')
print(f"Diagnosis: {result} (Confidence: {confidence:.2f})")