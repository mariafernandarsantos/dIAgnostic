import os
from dotenv import load_dotenv

load_dotenv()

"""
Configurações centralizadas e gerenciamento de variáveis de ambiente para a aplicação.
"""

load_dotenv()

# Configurações da API
API_TITLE = "dIAgnostic API"
API_DESCRIPTION = "API for medical diagnostics including pneumonia detection and diabetes prediction, enhanced with AI explanations"
API_VERSION = "1.5.0"

# Configurações do banco de dados
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./diagnostic.db")

# Configurações de segurança
SECRET_KEY = os.getenv("SECRET_KEY", "fallback-chave-fraca")
ALGORITHM = "HS256"
TOKEN_EXPIRATION_MINUTES = 60

# Configurações da API Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-1.5-pro"

# Caminhos para os modelos de ML
ML_MODELS_DIR = "ml_models"
PNEUMONIA_MODEL_PATH = f"{ML_MODELS_DIR}/pneumonia_detection_model.h5"
DIABETES_MODEL_PATH = f"{ML_MODELS_DIR}/diabetes_model.sav"
DIABETES_SCALER_PATH = f"{ML_MODELS_DIR}/diabetes_scaler.sav"

# Configurações CORS
CORS_ORIGINS = ["*"]
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOW_METHODS = ["*"]
CORS_ALLOW_HEADERS = ["*"]
