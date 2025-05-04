import requests
import json

url = "http://localhost:8000/predict/diabetes"
dados = {
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree": 0.627,
    "age": 50,
    "get_explanation": True
}

resposta = requests.post(url, json=dados)
print(json.dumps(resposta.json(), indent=2, ensure_ascii=False))
