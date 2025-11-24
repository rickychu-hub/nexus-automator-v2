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
    Writer 4.0: Genera Micro-Notas Tácticas pegadas a cada nodo complejo.
    Estrategia: Divide y vencerás.
    """
    logger.info("📝 Writer: Generando notas tácticas nodo a nodo...")
    
    # Lista de nodos que NO necesitan explicación (demasiado obvios o lógicos)
    skip_types = [
        "n8n-nodes-base.start", 
        "n8n-nodes-base.stickyNote", 
        "n8n-nodes-base.noOp"
    ]
    
    # Nodos que SIEMPRE necesitan ayuda (Apps externas)
    complex_nodes = [n for n in nodes if n.get("type") not in skip_types]
    
    new_notes = []

    # Procesamos nodo a nodo (esto consume más tiempo de IA, pero el resultado vale oro)
    # Para optimizar, podríamos hacerlo en una sola llamada "batch", pero nodo a nodo es más preciso.
    
    # Hacemos un prompt "Batch" inteligente para no llamar a la API 20 veces
    # Le pasamos la lista de nodos y pedimos un array de textos.
    
    nodes_summary = ""
    for i, n in enumerate(complex_nodes):
        # Le damos a la IA el ID temporal y los parámetros actuales para que vea qué falta
        params_snippet = str(n.get("parameters", {}))[:300] 
        nodes_summary += f"ID: {n.get('id')} | TIPO: {n.get('type')} | NOMBRE: {n.get('name')} | CONFIG_ACTUAL: {params_snippet}\n---\n"

    if not nodes_summary:
        return nodes

    prompt = (
        f"Eres un Instructor Experto en n8n. Tu alumno no sabe nada de automatización.\n"
        f"Analiza estos nodos y genera el texto para una 'Sticky Note' individual para cada uno.\n\n"
        f"**LISTA DE NODOS:**\n{nodes_summary}\n\n"
        f"**TU TAREA:**\n"
        f"Para cada nodo, genera un bloque de texto que siga ESTRICTAMENTE este formato:\n"
        f"NODE_ID: [El ID del nodo]\n"
        f"CONTENT:\n"
        f"🎯 **[Qué hace este nodo en 1 frase sencilla]**\n"
        f"⚠️ **TAREAS:**\n"
        f"- [Instrucción clara de qué credencial conectar]\n"
        f"- [Instrucción clara de qué campo rellenar y DÓNDE encontrar el dato]\n"
        f"⛔ **NO TOCAR:** La configuración avanzada.\n"
        f"END_CONTENT\n\n"
        f"**REGLAS:**\n"
        f"- Si el nodo es un 'Webhook' o 'Set', sé muy breve.\n"
        f"- Si es 'Google Sheets', 'Slack', 'Gmail', etc., explica dónde sacar los IDs.\n"
        f"- Usa emojis. Sé directo."
    )

    try:
        response = model.generate_content(prompt)
        full_text = clean_response(response.text)
        
        # Parseamos la respuesta para separar las notas
        # Buscamos bloques que empiecen por NODE_ID: y terminen en END_CONTENT
        pattern = r"NODE_ID:\s*(.*?)\s*CONTENT:\s*(.*?)\s*END_CONTENT"
        matches = re.findall(pattern, full_text, re.DOTALL)
        
        # Mapa rápido para encontrar nodos por ID
        node_map = {n['id']: n for n in nodes}

        for node_id, content in matches:
            target_node = node_map.get(node_id.strip())
            
            if target_node:
                # Posición: Ponemos la nota ENCIMA del nodo (eje Y - 200)
                # Ojo: Si el nodo no tiene posición aún (builder), usamos 0,0 y el Assembler lo arreglará
                original_pos = target_node.get("position", [0, 0])
                note_pos = [original_pos[0], original_pos[1] - 250] # 250px arriba

                sticky = {
                    "parameters": {
                        "content": content.strip(),
                        "height": 220,
                        "width": 300,
                        "color": 4 # Amarillo (Warning/Info)
                    },
                    "type": "n8n-nodes-base.stickyNote",
                    "typeVersion": 1,
                    "position": note_pos,
                    "id": f"note_{node_id}",
                    "name": f"Nota para {target_node.get('name')}"
                }
                new_notes.append(sticky)

    except Exception as e:
        logger.error(f"Error en Writer Batch: {e}")

    # Añadimos todas las notas generadas a la lista principal
    return nodes + new_notes