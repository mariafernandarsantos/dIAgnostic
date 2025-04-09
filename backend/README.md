# API de Detecção de Pneumonia

Uma API REST baseada em FastAPI para detectar pneumonia em imagens de raio-X do tórax usando um modelo de aprendizado profundo.

## Visão Geral

Esta API fornece um endpoint para enviar imagens de raio-X do tórax e receber predições sobre se a imagem mostra sinais de pneumonia ou pulmões normais. A API utiliza um modelo de rede neural convolucional treinado para fazer estas predições.

## Funcionalidades

- Envio de imagens de raio-X do tórax para detecção de pneumonia
- Obtenção de resultados de predição com níveis de confiança
- Construída com FastAPI para alto desempenho e recursos modernos de API
- Endpoint de verificação de saúde para verificar o status da API e do modelo

## Requisitos

- Python 3.8+
- FastAPI
- Uvicorn
- TensorFlow
- OpenCV
- NumPy

## Instalação

1. Clone este repositório:
```bash
git clone https://github.com/mariafernandarsantos/dIAgnostic.git
cd dIAgnostic
cd backend
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
# ou
pip install fastapi uvicorn python-multipart tensorflow opencv-python numpy
```

3. Baixe o arquivo do modelo:
   - Certifique-se de que o arquivo `pneumonia_detection_model.h5` esteja no diretório raiz do projeto
   - Se você precisar treinar seu próprio modelo, consulte o notebook de treinamento no repositório

## Uso

### Iniciando a API

Execute o seguinte comando no diretório do projeto:

```bash
uvicorn main:app --reload
# ou
python main.py
```

A API estará disponível em `http://localhost:8000`

### Endpoints da API

- `GET /`: Mensagem de boas-vindas e informações sobre endpoints disponíveis
- `GET /health`: Verificar se a API está em execução e se o modelo está carregado
- `POST /predict`: Enviar uma imagem e obter a predição de pneumonia

### Fazendo Predições

Você pode usar cURL para testar a API:

```bash
curl -X POST -F "file=@caminho/para/sua/imagem.jpg" http://localhost:8000/predict
```

Ou usar qualquer cliente HTTP como Postman, ou o seguinte código Python:

```python
import requests

url = "http://localhost:8000/predict"
caminho_imagem = "caminho/para/sua/imagem.jpg"

with open(caminho_imagem, "rb") as arquivo_imagem:
    arquivos = {"file": arquivo_imagem}
    resposta = requests.post(url, files=arquivos)

print(resposta.json())
```

### Formato da Resposta

A API retorna uma resposta JSON com a seguinte estrutura:

```json
{
  "filename": "exemplo.jpg",
  "diagnosis": "PNEUMONIA",
  "confidence": 0.9527,
  "raw_prediction": 0.9527
}
```

- `filename`: O nome do arquivo enviado
- `diagnosis`: "PNEUMONIA" ou "NORMAL"
- `confidence`: O nível de confiança (0-1) para a predição
- `raw_prediction`: A saída bruta do modelo

## Documentação da API

O FastAPI gera automaticamente documentação interativa da API:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## TODO produção
1. Ajuste nas configurações CORS na seção `app.add_middleware`
2. Configure autenticação adequada
3. Use um servidor ASGI de nível de produção como Gunicorn com workers Uvicorn
4. Implante atrás de um proxy reverso como Nginx
5. Use variáveis de ambiente para configuração
