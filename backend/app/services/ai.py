"""
Serviços de integração com IA para geração de explicações e chat.
"""
import base64
import google.generativeai as genai
from typing import Optional
from app.core.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas.prediction import DiabetesInput

# Configuração da API Gemini
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Modelo Gemini global
gemini_model = None
if GEMINI_API_KEY:
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)

async def get_pneumonia_explanation(image_bytes: bytes, diagnosis: str, confidence: float) -> str:
    """
    Obtém explicação de IA para diagnóstico de pneumonia em português.
    
    Args:
        image_bytes: Bytes da imagem de raio-X.
        diagnosis: Diagnóstico (PNEUMONIA ou NORMAL).
        confidence: Pontuação de confiança da predição.
        
    Returns:
        Explicação detalhada em português.
    """
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
        
        Mantenha sua resposta profissional, mas acessível para pacientes sem formação médica.
        Forneça uma análise abrangente em 3-4 parágrafos.
        
        IMPORTANTE: Responda COMPLETAMENTE em português do Brasil.
        """
        
        # Chamar API Gemini com a imagem
        response = await gemini_model.generate_content_async(
            [prompt, {"mime_type": "image/jpeg", "data": base64_image}]
        )
        
        return response.text
    except Exception as e:
        return f"Não foi possível gerar uma explicação detalhada: {str(e)}"

async def get_diabetes_explanation(input_data: DiabetesInput, diagnosis: str, probability: float) -> str:
    """
    Obtém explicação de IA para diagnóstico de diabetes em português.
    
    Args:
        input_data: Dados de entrada do paciente.
        diagnosis: Diagnóstico (POSITIVE ou NEGATIVE).
        probability: Probabilidade da predição.
        
    Returns:
        Explicação detalhada em português.
    """
    if gemini_model is None:
        return "Explicação detalhada não disponível. API Gemini não configurada."

    try:
        # Criar prompt para Gemini em português
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
        
        # Chamar API Gemini
        response = await gemini_model.generate_content_async(prompt)
        
        return response.text
    except Exception as e:
        return f"Não foi possível gerar uma explicação detalhada: {str(e)}"

async def chat_with_ai(message: str, context: Optional[dict] = None, history: Optional[list] = None) -> str:
    """
    Interage com a IA para responder perguntas sobre tópicos médicos.
    
    Args:
        message: Mensagem/pergunta do usuário.
        context: Informações de contexto opcionais (resultados de diagnóstico, etc.).
        history: Histórico opcional de conversas.
        
    Returns:
        Resposta da IA em português.
    """
    if gemini_model is None:
        return "Chat não disponível. API Gemini não configurada."

    try:
        # Preparar histórico de conversa se fornecido
        chat = gemini_model.start_chat(history=[])
        
        if history:
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    chat.send_message(content)
                else:
                    # Adicionar como resposta do modelo no histórico
                    chat.history.append({"role": "model", "parts": [content]})
        
        # Formatar o prompt com contexto se disponível
        prompt = message
        if context:
            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])
            prompt = f"""
            Pergunta do usuário: {message}
            
            Informações de contexto:
            {context_str}
            
            Por favor, forneça uma resposta útil, precisa e educativa à pergunta do usuário.
            Concentre-se em explicar conceitos médicos em termos simples, mantendo a precisão.
            
            IMPORTANTE: Responda COMPLETAMENTE em português do Brasil.
            """
        
        # Obter resposta do Gemini
        response = await chat.send_message_async(prompt)
        
        return response.text, id(chat)
    except Exception as e:
        return f"Erro ao processar chat: {str(e)}", None