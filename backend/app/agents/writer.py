# backend/app/agents/writer.py
import logging
import re
import google.generativeai as genai

logger = logging.getLogger(__name__)

def clean_response(text):
    """Limpia bloques de código y marcadores markdown."""
    text = re.sub(r'```[a-zA-Z]*', '', text).replace('```', '')
    return text.strip()

def agent_technical_writer(nodes, user_request, model):
    """
    Genera una Guía de Configuración paso a paso pensada para principiantes.
    """
    logger.info("📝 Writer: Generando instrucciones de configuración detalladas...")
    
    # Identificar nodos que requieren configuración manual (ignoramos los simples como Start)
    complex_nodes = [
        n for n in nodes 
        if n.get("type") not in ["n8n-nodes-base.start", "n8n-nodes-base.stickyNote"]
    ]
    
    if not complex_nodes:
        return nodes

    # Preparamos el resumen de lo que hace cada nodo para la IA
    nodes_context = ""
    for n in complex_nodes:
        # Extraemos parámetros clave para que la IA vea qué falta
        params = str(n.get("parameters", {}))[:200] 
        nodes_context += f"- Nodo '{n.get('name')}' ({n.get('type')}): {params}\n"

    prompt = (
        f"Actúa como un profesor experto en n8n enseñando a un alumno principiante.\n"
        f"Tu tarea es escribir una GUÍA DE CONFIGURACIÓN para este workflow.\n\n"
        f"**Objetivo del Workflow:** {user_request}\n"
        f"**Nodos a configurar:**\n{nodes_context}\n\n"
        f"**TU MISIÓN:**\n"
        f"Genera el texto para una Nota Adhesiva (Sticky Note) que guíe al usuario paso a paso para terminar de configurar el flujo. "
        f"Asume que el usuario NO sabe qué poner en los campos vacíos.\n\n"
        f"**REGLAS DE FORMATO:**\n"
        f"1. NO uses bloques de código (```). Usa texto limpio.\n"
        f"2. Usa emojis para guiar la vista.\n"
        f"3. ESTRUCTURA OBLIGATORIA:\n"
        f"   - 🏁 **PASO 1: Credenciales**\n"
        f"     (Lista qué cuentas debe conectar: Google, Slack, etc.)\n"
        f"   - 🛠️ **PASO 2: Configuración por Nodo**\n"
        f"     (Para cada nodo importante, di EXACTAMENTE qué campo rellenar. Ej: 'En Google Sheets, pon el ID de tu hoja en el campo Spreadsheet ID'.)\n"
        f"   - 🧪 **PASO 3: Cómo probarlo**\n"
        f"     (Explica cómo lanzar la primera prueba manual)\n"
    )

    try:
        response = model.generate_content(prompt)
        content = clean_response(response.text)
    except Exception as e:
        logger.error(f"Error en Writer: {e}")
        content = "⚠️ Error generando guía. Revisa cada nodo manualmente."

    # Configuración visual de la nota (Más ancha para leer bien las instrucciones)
    guide_note = {
        "parameters": {
            "content": content,
            "height": 600,
            "width": 500,
            "color": 3 # Verde (Éxito/Guía) o 4 (Amarillo)
        },
        "type": "n8n-nodes-base.stickyNote",
        "typeVersion": 1,
        "position": [-150, 100], # A la izquierda, bien visible
        "id": "nexus_guide_note",
        "name": "Guía de Configuración"
    }

    # Insertar al principio
    nodes.insert(0, guide_note)
    return nodes