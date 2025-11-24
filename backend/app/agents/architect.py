# backend/app/agents/architect.py
import logging
import json
import re
import google.generativeai as genai

logger = logging.getLogger(__name__)

def agent_architect(investigation_results, user_request, knowledge_base_memory, model):
    logger.info("🏛️ Iniciando Agente Arquitecto...")
    
    candidate_node_ids = investigation_results.get("candidate_nodes", [])
    case_studies = investigation_results.get("case_studies", []) # Estos son strings de patrones ahora

    # Preparamos contexto técnico del JSON en memoria
    candidate_details = []
    for nid in candidate_node_ids:
        node_info = knowledge_base_memory.get(nid.lower())
        if node_info:
            candidate_details.append({
                "nodeId": nid,
                "description": node_info.get('description', ''),
                "properties_schema": node_info.get('properties', {}) # Esquema real
            })

    prompt = (
        f"Eres el Arquitecto de Nexus OS. Diseña un workflow de n8n detallado.\n\n"
        f"**Petición:** \"{user_request}\"\n"
        f"**Patrones de Experiencia (Referencias):**\n{json.dumps(case_studies, indent=2)}\n\n"
        f"**Especificaciones Técnicas de Nodos:**\n{json.dumps(candidate_details, indent=2)}\n\n"
        f"**Instrucción:** Genera un ARRAY JSON con la lógica del flujo. Para cada paso incluye:\n"
        f"1. `nodeId`: ID exacto.\n"
        f"2. `parameters`: Configuración basada en el 'properties_schema'.\n"
        f"3. `purpose`: Explicación breve.\n"
        f"4. `branches`: (Solo si es IF) Objeto con claves 'true'/'false' conteniendo arrays de pasos.\n\n"
        f"Responde SOLO con el JSON."
    )

    try:
        response = model.generate_content(prompt)
        # Lógica robusta de extracción de JSON
        match = re.search(r'```json\s*(\[.*?\])\s*```', response.text, re.DOTALL)
        if not match:
             match = re.search(r'(\[.*\])', response.text, re.DOTALL)
        
        if match:
            plan = json.loads(match.group(1))
            logger.info("✅ Plan arquitectónico generado.")
            return plan
        else:
            logger.error("❌ El Arquitecto no devolvió un JSON Array válido.")
            return None
    except Exception as e:
        logger.error(f"❌ Error crítico en Arquitecto: {e}")
        return None