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
    
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1:
        return text[start:end+1]
    
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    return text.strip()

def agent_architect(investigation_results, user_request, knowledge_base_memory, model):
    logger.info("🏛️ Iniciando Agente Arquitecto (Estricto + Retry)...")
    
    candidate_node_ids = investigation_results.get("candidate_nodes", [])
    
    # Lista simple de IDs válidos para validación en el prompt
    valid_ids_list = [nid for nid in candidate_node_ids]

    candidate_details = []
    for nid in candidate_node_ids:
        node_info = knowledge_base_memory.get(nid.lower())
        if node_info:
            candidate_details.append({
                "nodeId_REAL": nid, # Enfatizamos que este es el REAL
                "description": node_info.get('description', ''),
                "properties_hint": "Usa properties estándar." 
            })

    base_prompt = (
        f"Eres el Arquitecto de Nexus OS. Diseña un workflow de n8n.\n\n"
        f"**Petición:** \"{user_request}\"\n"
        f"**CATÁLOGO DE NODOS VÁLIDOS (NO INVENTES NADA FUERA DE AQUÍ):**\n"
        f"{json.dumps(candidate_details, indent=2)}\n\n"
        f"**REGLAS DE ORO (CRÍTICO):**\n"
        f"1. **PROHIBIDO INVENTAR TIPOS:** Usa EXCLUSIVAMENTE los valores del campo `nodeId_REAL` (ej: 'n8n-nodes-base.webhook'). NO uses nombres como 'webhook_trigger' o 'router'.\n"
        f"2. **JSON PURO:** Devuelve SOLO el Array JSON. Sin texto, sin markdown.\n"
        f"3. **RAMAS:** Usa la estructura `branches` para IF y Switch.\n\n"
        f"Estructura de Salida:\n"
        f"[\n"
        f"  {{\n"
        f"    \"nodeId\": \"n8n-nodes-base.webhook\",\n"
        f"    \"purpose\": \"Recibir datos\",\n"
        f"    \"parameters\": {{ ... }},\n"
        f"    \"branches\": {{ ... }} \n"
        f"  }}\n"
        f"]"
    )

    # Lógica de Reintento (Retry)
    try:
        response = model.generate_content(base_prompt)
        cleaned_text = clean_json_text(response.text)
        plan = json.loads(cleaned_text)
        
        # Validación Rápida: ¿Ha inventado nodos?
        for node in plan:
            if "n8n-nodes-base" not in node.get("nodeId", ""):
                raise ValueError(f"Nodo inválido detectado: {node.get('nodeId')}")
        
        return plan

    except Exception as e1:
        logger.warning(f"⚠️ Arquitecto Intento 1 falló ({e1}). Reintentando con corrección...")
        
        try:
            retry_prompt = (
                f"Tu respuesta anterior falló o contenía nodos inventados. ERROR: {str(e1)}\n"
                f"CORRIGE el JSON.\n"
                f"IMPORTANTE: Usa SOLO IDs que empiecen por 'n8n-nodes-base.'.\n"
                f"Lista permitida: {json.dumps(valid_ids_list)}"
            )
            time.sleep(1)
            response_retry = model.generate_content(base_prompt + "\n\n" + retry_prompt)
            cleaned_text_retry = clean_json_text(response_retry.text)
            
            return json.loads(cleaned_text_retry)
            
        except Exception as e2:
            logger.error(f"❌ Error crítico en Arquitecto tras reintento: {e2}")
            return None