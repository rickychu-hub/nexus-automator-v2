# backend/app/agents/architect.py
import logging
import json
import re
import time
import google.generativeai as genai

logger = logging.getLogger(__name__)

def clean_json_text(text):
    """
    Limpieza agresiva para extraer JSON de respuestas con ruido.
    """
    if not text: return ""
    
    # 1. Intentar encontrar el bloque JSON más externo [...]
    # Esto busca el primer corchete de apertura y el último de cierre
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1:
        candidate = text[start:end+1]
        return candidate
    
    # Fallback: quitar markdown si no encontró corchetes claros
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def agent_architect(investigation_results, user_request, knowledge_base_memory, model):
    logger.info("🏛️ Iniciando Agente Arquitecto (Con Auto-Corrección)...")
    
    candidate_node_ids = investigation_results.get("candidate_nodes", [])
    case_studies = investigation_results.get("case_studies", []) 

    candidate_details = []
    for nid in candidate_node_ids:
        node_info = knowledge_base_memory.get(nid.lower())
        if node_info:
            candidate_details.append({
                "nodeId": nid,
                "description": node_info.get('description', ''),
                "properties_hint": "Usa properties estándar." 
            })

    base_prompt = (
        f"Eres el Arquitecto de Nexus OS. Diseña un workflow de n8n detallado.\n\n"
        f"**Petición:** \"{user_request}\"\n"
        f"**Nodos Disponibles:**\n{json.dumps(candidate_details, indent=2)}\n\n"
        f"**Instrucción CRÍTICA:**\n"
        f"Genera un ARRAY JSON válido. NO escribas texto antes ni después. NO uses comentarios.\n"
        f"Estructura requerida:\n"
        f"[\n"
        f"  {{\n"
        f"    \"nodeId\": \"...\",\n"
        f"    \"purpose\": \"...\",\n"
        f"    \"parameters\": {{ ... }},\n"
        f"    \"branches\": {{ \"true\": [...], \"false\": [...] }} \n"
        f"  }}\n"
        f"]"
    )

    # INTENTO 1
    try:
        response = model.generate_content(base_prompt)
        cleaned_text = clean_json_text(response.text)
        return json.loads(cleaned_text)
    except Exception as e1:
        logger.warning(f"⚠️ Arquitecto Intento 1 falló: {e1}. Iniciando reintento de corrección...")
        
        # INTENTO 2 (Auto-Corrección)
        try:
            # Le pasamos el error a la IA para que se corrija
            retry_prompt = (
                f"Tu respuesta anterior generó un error de sintaxis JSON. \n"
                f"Por favor, genera SOLO el código JSON corregido, sin explicaciones.\n"
                f"Asegúrate de cerrar todos los corchetes '[' y llaves '{{'."
            )
            
            # Pequeña pausa para no saturar
            time.sleep(1)
            
            # En chat mode para mantener contexto es mejor, pero aquí hacemos nueva llamada simple
            # concatenando para dar contexto
            response_retry = model.generate_content(base_prompt + "\n\n" + retry_prompt)
            cleaned_text_retry = clean_json_text(response_retry.text)
            
            plan = json.loads(cleaned_text_retry)
            logger.info("✅ Arquitecto recuperado en Intento 2.")
            return plan
            
        except Exception as e2:
            logger.error(f"❌ Error crítico en Arquitecto tras reintento: {e2}")
            logger.error(f"Texto fallido final: {response_retry.text[:200]}...")
            return None