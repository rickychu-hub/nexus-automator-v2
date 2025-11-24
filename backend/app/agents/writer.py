# backend/app/agents/writer.py
import logging
import re
import google.generativeai as genai

logger = logging.getLogger(__name__)

def clean_response(text):
    text = re.sub(r'```[a-zA-Z]*', '', text).replace('```', '')
    return text.strip()

def agent_technical_writer(nodes, user_request, model):
    """
    Writer 5.0: Notas Tácticas vinculadas por ID para posicionamiento inteligente.
    """
    logger.info("📝 Writer: Generando notas tácticas...")
    
    # Ignoramos nodos simples para no llenar la pantalla de ruido
    skip_types = ["n8n-nodes-base.start", "n8n-nodes-base.stickyNote", "n8n-nodes-base.noOp"]
    complex_nodes = [n for n in nodes if n.get("type") not in skip_types]
    
    new_notes = []
    
    # Resumen para la IA
    nodes_summary = ""
    for n in complex_nodes:
        params_snippet = str(n.get("parameters", {}))[:300] 
        nodes_summary += f"ID: {n.get('id')} | TIPO: {n.get('type')} | NOMBRE: {n.get('name')} | PARAMS: {params_snippet}\n---\n"

    if not nodes_summary:
        return nodes

    prompt = (
        f"Eres un Instructor de n8n. Genera el texto para Sticky Notes individuales.\n"
        f"**NODOS:**\n{nodes_summary}\n\n"
        f"**FORMATO OBLIGATORIO POR NODO:**\n"
        f"NODE_ID: [ID del nodo]\n"
        f"CONTENT:\n"
        f"📌 **Guía para: [Nombre del Nodo]**\n"
        f"🎯 [Qué hace en 1 frase]\n"
        f"⚠️ **ACCIÓN REQUERIDA:**\n"
        f"- [Instrucción credencial]\n"
        f"- [Instrucción campo vacío]\n"
        f"END_CONTENT\n"
    )

    try:
        response = model.generate_content(prompt)
        full_text = clean_response(response.text)
        
        pattern = r"NODE_ID:\s*(.*?)\s*CONTENT:\s*(.*?)\s*END_CONTENT"
        matches = re.findall(pattern, full_text, re.DOTALL)
        
        node_map = {n['id']: n for n in nodes}

        for node_id, content in matches:
            target_node = node_map.get(node_id.strip())
            if target_node:
                # CREAMOS LA NOTA PERO NO LE DAMOS POSICIÓN FINAL AÚN.
                # El "Assembler" se encargará de pegarla a su padre.
                sticky = {
                    "parameters": {
                        "content": content.strip(),
                        "height": 240,
                        "width": 350,
                        "color": 4 # Amarillo
                    },
                    "type": "n8n-nodes-base.stickyNote",
                    "typeVersion": 1,
                    "position": [0, 0], # Temporal
                    "id": f"note_{target_node['id']}", # ID vinculado clave
                    "name": f"Nota: {target_node['name']}"
                }
                new_notes.append(sticky)

    except Exception as e:
        logger.error(f"Error en Writer: {e}")

    return nodes + new_notes