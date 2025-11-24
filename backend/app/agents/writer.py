# backend/app/agents/writer.py
import logging
import google.generativeai as genai

logger = logging.getLogger(__name__)

def agent_technical_writer(nodes, user_request, model):
    """
    Genera Sticky Notes con documentación para el usuario.
    """
    logger.info("📝 Writer: Redactando manual de vuelo...")
    
    # Filtramos nodos que no sean notas ya existentes
    target_nodes = [n for n in nodes if "stickyNote" not in n.get("type", "")]
    
    if not target_nodes:
        return nodes

    # Prompt eficiente en lote (una sola llamada para todo el flujo)
    # Nota: Si el flujo es muy largo, en el futuro se debería hacer por lotes.
    node_summaries = "\n".join([f"- {n.get('name')} ({n.get('type')}): {n.get('purpose')}" for n in target_nodes])

    prompt = (
        f"Actúa como redactor técnico de n8n. Genera una nota explicativa BREVE para este flujo:\n"
        f"Contexto: {user_request}\n"
        f"Nodos:\n{node_summaries}\n\n"
        f"Tu tarea: Genera un texto para una sola Sticky Note grande que resuma cómo funciona este flujo y qué credenciales debe revisar el usuario.\n"
        f"Usa formato Markdown simple."
    )

    try:
        response = model.generate_content(prompt)
        note_content = response.text.strip()
    except Exception as e:
        logger.error(f"Error en Writer: {e}")
        note_content = "Revisa la configuración de cada nodo y tus credenciales."

    # Crear el nodo Sticky Note
    sticky_note = {
        "parameters": {
            "content": note_content,
            "height": 400,
            "width": 500,
            "color": 4 # Amarillo n8n
        },
        "type": "n8n-nodes-base.stickyNote",
        "typeVersion": 1,
        "position": [100, 100], # Al principio
        "id": "nexus_instruction_note",
        "name": "Instrucciones Nexus"
    }

    nodes.insert(0, sticky_note)
    return nodes