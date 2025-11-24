# backend/app/agents/architect.py
import logging
import json
import google.generativeai as genai

logger = logging.getLogger(__name__)

def agent_architect(investigation_results, user_request, knowledge_base_memory, model):
    logger.info("🏛️ Iniciando Agente Arquitecto (V11 - Schema Enforced)...")
    
    candidate_node_ids = investigation_results.get("candidate_nodes", [])
    
    candidate_details = []
    for nid in candidate_node_ids:
        node_info = knowledge_base_memory.get(nid.lower())
        if node_info:
            candidate_details.append({
                "nodeId": nid,
                "desc": node_info.get('description', '')[:100]
            })

    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    # PROMPT MEJORADO CON EJEMPLO DE ESTRUCTURA DE ÁRBOL
    prompt = (
        f"Eres el Arquitecto de Nexus OS. Diseña un workflow de n8n para: \"{user_request}\"\n\n"
        f"**NODOS DISPONIBLES:**\n{json.dumps(candidate_details, indent=2)}\n\n"
        f"**REGLA CRÍTICA DE ESTRUCTURA (ÁRBOL):**\n"
        f"El JSON debe representar un ÁRBOL lógico. Si usas un IF o SWITCH, debes usar la propiedad `branches`.\n\n"
        f"**EJEMPLO DE FORMATO OBLIGATORIO:**\n"
        f"[\n"
        f"  {{\n"
        f"    \"nodeId\": \"n8n-nodes-base.webhook\",\n"
        f"    \"parameters\": {{ ... }}\n"
        f"  }},\n"
        f"  {{\n"
        f"    \"nodeId\": \"n8n-nodes-base.switch\",\n"
        f"    \"parameters\": {{ ... }},\n"
        f"    \"branches\": {{\n"
        f"      \"EU\": [ {{ \"nodeId\": \"n8n-nodes-base.set\", ... }}, {{ \"nodeId\": \"n8n-nodes-base.slack\", ... }} ],\n"
        f"      \"NA\": [ {{ \"nodeId\": \"n8n-nodes-base.set\", ... }} ]\n"
        f"    }}\n"
        f"  }}\n"
        f"]\n\n"
        f"**TU TAREA:** Genera el JSON del workflow completo siguiendo esta estructura anidada."
    )

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        plan = json.loads(response.text)
        
        # Validación: Si es un switch, DEBE tener branches
        for node in plan:
            if "switch" in node.get("nodeId", "") and "branches" not in node:
                logger.warning("⚠️ El Arquitecto generó un Switch sin ramas. Esto fallará.")
                
        return plan

    except Exception as e:
        logger.error(f"❌ Error en Arquitecto V11: {e}", exc_info=True)
        return None