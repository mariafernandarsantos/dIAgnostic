"""
Funções para processamento de imagens e outras utilidades.
"""
import numpy as np
import cv2
from typing import Tuple

def preprocess_image(image_bytes: bytes) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pré-processa a imagem a partir de bytes para predição.
    
    Args:
        image_bytes: Bytes da imagem a ser processada.
        
    Returns:
        Uma tupla contendo a imagem processada para o modelo e a imagem redimensionada original.
        
    Raises:
        ValueError: Se ocorrer um erro ao processar a imagem.
    """
    try:
        # Converter bytes para array numpy
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decodificar imagem como colorida (3 canais)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Não foi possível decodificar a imagem")

        # Redimensionar para 256x256
        img = cv2.resize(img, (256, 256))

        # Normalizar valores de pixel
        normalized_img = img / 255.0

        # Remodelar para o modelo (tamanho de lote 1)
        model_input = np.reshape(normalized_img, (1, 256, 256, 3))

        # Retornar imagem processada para o modelo e imagem redimensionada original
        return model_input, cv2.resize(cv2.imdecode(nparr, cv2.IMREAD_COLOR), (256, 256))
    except Exception as e:
        raise ValueError(f"Erro ao pré-processar a imagem: {e}")