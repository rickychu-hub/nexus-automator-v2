import logging
import json
import os
import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)

# Configuración de Gemini
if settings.GOOGLE_API_KEY:
    genai.configure(api_key=settings.GOOGLE_API_KEY)

def analyze_error(error_message: str, workflow_context: dict = None) -> dict:
    """
    Analiza un error de n8n usando IA y sugiere una solución estructurada.
    """
    if not settings.GOOGLE_API_KEY:
        logger.warning("🚫 Google API Key no configurada. Saltando análisis de error.")
        return {"diagnosis": "IA no disponible", "fix": "Configura GOOGLE_API_KEY"}

    try:
        model = genai.GenerativeModel('gemini-2.0-flash-exp')
        
        context_str = json.dumps(workflow_context, indent=2) if workflow_context else "No context provided"
        
        prompt = (
            f"Eres un experto en n8n y SRE (Site Reliability Engineering). Analiza este error:\n"
            f"ERROR: \"{error_message}\"\n\n"
            f"CONTEXTO DEL WORKFLOW:\n{context_str}\n\n"
            f"Tarea: Explica la causa raíz en una frase simple y da 3 pasos numerados para solucionarlo.\n"
            f"FORMATO JSON:\n"
            f"{{ \"diagnosis\": \"...\", \"suggested_fix\": \"1. ... 2. ... 3. ...\" }}"
        )

        response = model.generate_content(prompt)
        text = response.text.strip()
        
        # Limpieza de markdown json si existe
        if text.startswith("```json"):
            text = text.replace("```json", "").replace("```", "")
        
        return json.loads(text)

    except Exception as e:
        logger.error(f"Error analizando error con IA: {e}")
        return {"diagnosis": "Error interno IA", "fix": "Check logs"}
