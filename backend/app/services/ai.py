import base64
import google.generativeai as genai
from app.core.config import GEMINI_API_KEY
from typing import List, Optional
import json

# Configurar Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

async def chat_with_ai(
    message: str,
    chat_history: List = None,
    prediction_history: List = None,
    user_context = None
) -> str:
    """
    Assistente de IA para apoio à decisão clínica médica.
    """
    try:
        # Contexto médico profissional
        medical_context = """
        Você é um sistema de apoio à decisão clínica para médicos, especializado em:
        - Análise de radiografias torácicas para pneumonia
        - Avaliação de fatores de risco para diabetes mellitus
        
        Forneça:
        1. Análises concisas baseadas em evidências
        2. Diferenciais diagnósticos relevantes
        3. Sugestões de exames complementares quando aplicável
        4. Referências a guidelines clínicos atualizados
        """

        # Construir histórico de casos
        case_history = []
        if prediction_history:
            case_history.append("\n**Histórico de Casos Recentes:**")
            for pred in prediction_history[-3:]:
                case_history.append(
                    f"- Caso {pred.prediction_type.upper()}: {pred.result} "
                    f"(Confiança: {pred.confidence}%)"
                )

        # Construir diálogo anterior
        dialog_history = []
        if chat_history:
            dialog_history.append("\n**Diálogo Anterior:**")
            for chat in reversed(chat_history[-5:]):
                dialog_history.append(f"**Médico:** {chat.message}")
                dialog_history.append(f"**Sistema:** {chat.response}")

        prompt = f"""
        {medical_context}
        
        {''.join(case_history)}
        {''.join(dialog_history)}
        
        **Consulta Atual:**
        {message}
        
        **Orientações para Resposta:**
        - Priorize informações clinicamente acionáveis
        - Destaque achados críticos em negrito
        - Inclua ESCORE de probabilidade clínica quando relevante
        - Sugira protocolos de conduta baseados em:
          * Diretrizes SBPT para pneumonias
          * Guidelines ADA para diabetes
        - Mantenha respostas em tópicos numerados
        - Use linguagem técnica apropriada para médicos
        """

        response = gemini_model.generate_content(prompt)
        return response.text
        
    except Exception as e:
        return "Sistema temporariamente indisponível. Por favor, tente novamente."

async def get_pneumonia_explanation(image_bytes: bytes, diagnosis: str, confidence: float) -> str:
    """Gera explicação detalhada para predição de pneumonia."""
    if gemini_model is None:
        return "Explicação detalhada não disponível. API Gemini não configurada."
    try:
        # Converter imagem para base64 para Gemini
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Criar prompt para Gemini em português
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
        
        Mantenha sua resposta profissional, mas acessível tanto para médicos especializados para auxiliar eles e para pacientes sem formação médica.
        Forneça uma análise abrangente em 3-4 parágrafos.
        
        IMPORTANTE: Responda COMPLETAMENTE em português do Brasil.
        """
        
        # Chamar API Gemini com a imagem
        response = await gemini_model.generate_content_async(
            [prompt, {"mime_type": "image/jpeg", "data": base64_image}]
        )
        
        return response.text
    except Exception as e:
        return "Explicação não disponível no momento."

async def get_diabetes_explanation(input_data, diagnosis: str, probability: float) -> str:
    """Relatório de avaliação de risco para diabetes"""
    if gemini_model is None:
        return "Explicação detalhada não disponível. API Gemini não configurada."
    try:
        prompt = f"""
        **Avaliação de Risco para Diabetes Mellitus**
        
        **Dados do Paciente:**
        - Idade: {input_data.age} anos
        - Glicemia: {input_data.glucose} mg/dL
        - IMC: {input_data.bmi} kg/m²
        - Outros fatores de risco: [listar]
        
        **Resultado da Triagem:**
        - Risco calculado: {probability:.2%}
        - Classificação: {diagnosis}
        
        **Análise Clínica:**
        1. **Critérios ADA Preenchidos:**
        - [Listar critérios atendidos]
        
        2. **Recomendações:**
        - Exames laboratoriais complementares:
            * HbA1c
            * Curva glicêmica
            * Perfil lipídico
        
        3. **Conduta Sugerida:**
        - Follow-up em [semanas/meses]
        - Encaminhamentos sugeridos:
            * Endocrinologia
            * Nutrição
        
        4. **Orientações para o Paciente:**
        - Monitoramento domiciliar
        - Sinais de alerta para emergências
        - Modificações no estilo de vida
        """
        
        response = await gemini_model.generate_content_async(prompt)
        return format_medical_response(response.text)
    except Exception as e:
        return "Explicação não disponível no momento."

def format_medical_response(text: str) -> str:
    """Formata a resposta para melhor legibilidade médica"""
    sections = [
        "DESCRIÇÃO TÉCNICA",
        "ANÁLISE COMPARATIVA",
        "SUGESTÕES",
        "COMENTÁRIOS ADICIONAIS",
        "RECOMENDAÇÕES"
    ]
    
    for section in sections:
        text = text.replace(section, f"\n**{section}**\n")
    
    return text