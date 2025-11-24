# backend/app/agents/architect.py
import logging
import json
import google.generativeai as genai

logger = logging.getLogger(__name__)

def agent_architect(investigation_results, user_request, knowledge_base_memory, model):
    logger.info("🏛️ Iniciando Agente Arquitecto (V12 - Hybrid Master)...")
    
    candidate_node_ids = investigation_results.get("candidate_nodes", [])
    
    # Lista de IDs válidos para que la IA los tenga a mano
    valid_ids_str = ", ".join([f"'{nid}'" for nid in candidate_node_ids])

    candidate_details = []
    for nid in candidate_node_ids:
        node_info = knowledge_base_memory.get(nid.lower())
        if node_info:
            candidate_details.append({
                "nodeId_REAL": nid, # Enfatizamos que este es el ID real
                "desc": node_info.get('description', '')[:100]
            })

    generation_config = {
        "temperature": 0.1,
        "top_p": 0.95,
        "max_output_tokens": 8192,
        "response_mime_type": "application/json",
    }

    prompt = (
        f"Eres el Arquitecto de Nexus OS. Diseña un workflow de n8n para: \"{user_request}\"\n\n"
        f"**NODOS DISPONIBLES (CATÁLOGO ESTRICTO):**\n{json.dumps(candidate_details, indent=2)}\n\n"
        f"**REGLA #1 (PROHIBIDO INVENTAR):**\n"
        f"Usa EXCLUSIVAMENTE los valores de `nodeId_REAL` de la lista anterior. \n"
        f"NO uses nombres inventados como 'webhookTrigger' o 'router'. Usa 'n8n-nodes-base.webhook', 'n8n-nodes-base.switch', etc.\n\n"
        f"**REGLA #2 (ESTRUCTURA DE ÁRBOL):**\n"
        f"El JSON debe tener estructura anidada. Si usas IF o SWITCH, usa la propiedad `branches`.\n\n"
        f"**EJEMPLO OBLIGATORIO:**\n"
        f"[\n"
        f"  {{\n"
        f"    \"nodeId\": \"n8n-nodes-base.webhook\",\n"
        f"    \"parameters\": {{ ... }}\n"
        f"  }},\n"
        f"  {{\n"
        f"    \"nodeId\": \"n8n-nodes-base.switch\",\n"
        f"    \"parameters\": {{ ... }},\n"
        f"    \"branches\": {{\n"
        f"      \"EU\": [ {{ \"nodeId\": \"n8n-nodes-base.set\", ... }} ],\n"
        f"      \"NA\": [ ... ]\n"
        f"    }}\n"
        f"  }}\n"
        f"]"
    )

    try:
        response = model.generate_content(prompt, generation_config=generation_config)
        plan = json.loads(response.text)
        
        # --- VALIDACIÓN DE EMERGENCIA ---
        # Si detectamos nodos inventados, forzamos un error para no entregar basura
        for node in plan:
            nid = node.get("nodeId", "")
            if not nid.startswith("n8n-nodes-base."):
                logger.error(f"❌ Arquitecto inventó un nodo: {nid}")
                # Intentar corrección simple: si inventó 'webhookTrigger', buscar 'n8n-nodes-base.webhook' en la lista válida
                for valid in candidate_node_ids:
                    if "webhook" in nid.lower() and "webhook" in valid.lower():
                        node["nodeId"] = valid
                        logger.info(f"🔧 Auto-corregido a: {valid}")
                        break
                    elif "switch" in nid.lower() and "switch" in valid.lower():
                        node["nodeId"] = valid
                        logger.info(f"🔧 Auto-corregido a: {valid}")
                        break
                    elif "router" in nid.lower() and "switch" in valid.lower(): # Router suele ser Switch
                        node["nodeId"] = valid
                        logger.info(f"🔧 Auto-corregido a: {valid}")
                        break
                
        return plan

    except Exception as e:
        logger.error(f"❌ Error en Arquitecto V12: {e}", exc_info=True)
        return None