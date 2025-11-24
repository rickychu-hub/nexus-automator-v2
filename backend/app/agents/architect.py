# backend/app/agents/architect.py
import logging
import json
import google.generativeai as genai

logger = logging.getLogger(__name__)

def agent_architect(investigation_results, user_request, knowledge_base_memory, model):
    logger.info("🏛️ Iniciando Agente Arquitecto (V10 - Native JSON Mode)...")
    
    candidate_node_ids = investigation_results.get("candidate_nodes", [])
    
    # Preparar contexto técnico
    candidate_details = []
    for nid in candidate_node_ids:
        node_info = knowledge_base_memory.get(nid.lower())
        if node_info:
            candidate_details.append({
                "nodeId": nid,
                "desc": node_info.get('description', '')[:100], # Resumido
                "props_hint": "Usa properties estándar." 
            })

    # CONFIGURACIÓN DETERMINISTA Y NATIVA
    # Forzamos a Gemini a devolver JSON puro sin markdown ni texto extra
    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192, # Suficiente espacio para workflows gigantes
        "response_mime_type": "application/json", # ¡LA CLAVE!
    }

    prompt = (
        f"Eres el Arquitecto de Nexus OS. Diseña un workflow de n8n para esta petición:\n"
        f"\"{user_request}\"\n\n"
        f"**NODOS DISPONIBLES:**\n{json.dumps(candidate_details, indent=2)}\n\n"
        f"**REGLAS OBLIGATORIAS:**\n"
        f"1. Usa SOLO los `nodeId` proporcionados. No inventes nombres.\n"
        f"2. Para nodos IF/SWITCH, usa la estructura `branches` con claves claras (ej: 'true', 'false' o nombres de ruta).\n"
        f"3. Devuelve UNICAMENTE una lista de objetos JSON.\n\n"
        f"**SCHEMA DE RESPUESTA (Array):**\n"
        f"[\n"
        f"  {{\n"
        f"    \"nodeId\": \"n8n-nodes-base.webhook\",\n"
        f"    \"purpose\": \"Explicación breve\",\n"
        f"    \"parameters\": {{ \"path\": \"...\", ... }},\n"
        f"    \"branches\": {{ ... }} \n"
        f"  }}\n"
        f"]"
    )

    try:
        # Llamada con configuración nativa
        response = model.generate_content(
            prompt, 
            generation_config=generation_config
        )
        
        # Al usar response_mime_type, el texto ya es JSON válido
        plan = json.loads(response.text)
        
        # Validación rápida de seguridad
        if not isinstance(plan, list):
            # A veces devuelve {"nodes": [...]}, normalizamos
            if isinstance(plan, dict) and "nodes" in plan:
                return plan["nodes"]
            elif isinstance(plan, dict):
                return [plan] # Si devolvió un solo nodo
                
        logger.info(f"✅ Arquitecto generó plan con {len(plan)} pasos.")
        return plan

    except Exception as e:
        logger.error(f"❌ Error en Arquitecto V10: {e}", exc_info=True)
        # Fallback de emergencia: devolver lo que se pueda
        return None