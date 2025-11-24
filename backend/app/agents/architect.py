# backend/app/agents/architect.py
import logging
import json
import re
import google.generativeai as genai

logger = logging.getLogger(__name__)

def clean_json_text(text):
    """
    Limpia la respuesta de la IA para extraer solo el JSON válido.
    Elimina bloques de código Markdown y busca el primer '[' y último ']'.
    """
    if not text:
        return ""
    
    # 1. Eliminar marcadores de Markdown (```json ... ```)
    text = re.sub(r'```json\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'```\s*', '', text)
    
    # 2. Buscar el array JSON explícitamente
    start = text.find('[')
    end = text.rfind(']')
    
    if start != -1 and end != -1:
        return text[start:end+1]
    
    return text.strip()

def agent_architect(investigation_results, user_request, knowledge_base_memory, model):
    logger.info("🏛️ Iniciando Agente Arquitecto...")
    
    candidate_node_ids = investigation_results.get("candidate_nodes", [])
    case_studies = investigation_results.get("case_studies", []) 

    # Preparamos contexto técnico del JSON en memoria (Limitado para no saturar token limit)
    candidate_details = []
    for nid in candidate_node_ids:
        node_info = knowledge_base_memory.get(nid.lower())
        if node_info:
            candidate_details.append({
                "nodeId": nid,
                "description": node_info.get('description', ''),
                # Simplificamos properties para ahorrar tokens y reducir errores de sintaxis
                "properties_hint": "Usa las propiedades estándar de este nodo." 
            })

    prompt = (
        f"Eres el Arquitecto de Nexus OS. Diseña un workflow de n8n detallado.\n\n"
        f"**Petición:** \"{user_request}\"\n"
        f"**Nodos Disponibles:**\n{json.dumps(candidate_details, indent=2)}\n\n"
        f"**Instrucción CRÍTICA:**\n"
        f"Genera un ARRAY JSON válido. NO escribas texto antes ni después. NO uses comentarios //.\n"
        f"Asegúrate de cerrar todas las comillas y corchetes.\n\n"
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

    try:
        # Generar contenido
        response = model.generate_content(prompt)
        raw_text = response.text
        
        # Limpieza quirúrgica
        cleaned_text = clean_json_text(raw_text)
        
        # Intentar parsear
        try:
            plan = json.loads(cleaned_text)
            logger.info("✅ Plan arquitectónico generado y parseado correctamente.")
            return plan
        except json.JSONDecodeError as e:
            logger.error(f"❌ Error de Sintaxis JSON en Arquitecto: {e}")
            logger.debug(f"Texto problemático: {cleaned_text[:500]}...") # Log parcial para debug
            return None
            
    except Exception as e:
        logger.error(f"❌ Error crítico en Arquitecto (General): {e}", exc_info=True)
        return None