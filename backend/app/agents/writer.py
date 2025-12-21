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
    Writer 5.0: Notas Tácticas + Guía de Configuración (Spec-Sheet).
    """
    logger.info("📝 Writer: Generando notas tácticas y Spec-Sheet...")
    
    # Ignoramos nodos simples para no llenar la pantalla de ruido
    skip_types = ["n8n-nodes-base.start", "n8n-nodes-base.stickyNote", "n8n-nodes-base.noOp"]
    complex_nodes = [n for n in nodes if n.get("type") not in skip_types]
    
    new_notes = []
    guide_markdown = "" # <--- Nueva variable para la guía
    
    # Resumen para la IA
    nodes_summary = ""
    for n in complex_nodes:
        params_snippet = str(n.get("parameters", {}))[:300] 
        # Detectar si tiene pinnedData (Mock Data)
        has_mock_data = "True" if n.get("pinnedData") else "False"
        nodes_summary += f"ID: {n.get('id')} | TIPO: {n.get('type')} | NOMBRE: {n.get('name')} | MOCK_DATA: {has_mock_data} | PARAMS: {params_snippet}\n---\n"

    if not nodes_summary:
        return nodes, "" # <--- Retornamos tupla vacía

    prompt = (
        f"Eres un n8n Solutions Engineer experto. Genera dos cosas:\n"
        f"1. Contenido para Sticky Notes (Protocolo Rápido).\n"
        f"2. Una Guía de Configuración paso a paso (Spec-Sheet) para el usuario.\n\n"
        f"**OBJETIVO:** Que el usuario configure el workflow sin dudas.\n\n"
        f"**NODOS A DOCUMENTAR:**\n{nodes_summary}\n\n"
        f"**--- SECCIÓN 1: STICKY NOTES ---**\n"
        f"Genera bloques con este formato EXACTO:\n"
        f"NODE_ID: [ID del nodo]\n"
        f"Note_Content:\n"
        f"## 🛠️ Configuración: [Nombre del Nodo]\n"
        f"([Frase corta sobre qué hace])\n\n"
        f"### Estado:\n"
        f"🔴 **Requiere Credencial** (Si es API/Service)\n"
        f"🟡 **Revisar Parámetro** (Si tiene campos vacíos)\n"
        f"🟢 **Listo** (Si parece completo)\n\n"
        f"### ⚡ Quick actions:\n"
        f"- **Data Path:** `[EXPRESIÓN EXACTA]` (ej: {{{{ $json.body.data }}}})\n"
        f"- **Wiring Check:** Verificar Output Key -> Input Key.\n"
        f"End_Note_Content\n\n"
        f"**--- SECCIÓN 2: SPEC-SHEET (Manual de Configuración) ---**\n"
        f"Genera un bloque MARKDOWN ÚNICO llamado 'SPEC_SHEET_START' hasta 'SPEC_SHEET_END'.\n"
        f"Sigue ESTRICTAMENTE esta estructura y estilo:\n\n"
        f"### 1. 🔐 Credenciales Requeridas\n"
        f"Listar cada servicio externo (OpenAI, Slack, Google, etc) usado en el workflow.\n"
        f"* **[Servicio]:** [Instrucción breve para obtener credencial/token].\n"
        f"    * *Ejemplo:* **Slack Token:** Tipo 'Bot User OAuth Token'. Scopes: `chat:write`.\n\n"
        f"### 2. 🎛️ Parámetros a Configurar\n"
        f"Listar los campos CLAVE que el usuario DEBE rellenar (IDs de carpetas, correos, canales).\n"
        f"* **Nodo [Nombre]:** Campo `[Campo]`. [Instrucción 'copy-paste' clara].\n"
        f"    * *Ejemplo:* **Nodo Notion:** Campo `Database ID`. Copia el ID de la URL `notion.so/my-db-[ID]`.\n\n"
        f"### 3. 🚀 Cómo probarlo\n"
        f"1. Haz clic en 'Execute Workflow'.\n"
        f"2. [Paso específico según el trigger: enviar correo, llenar form, etc].\n"
        f"3. [Resultado esperado: confirmar recepción en Slack/Email].\n"
        f"4. (Si hay Mock Data) *Nota: Este workflow incluye datos de prueba para validación inmediata.*\n"
    )

    try:
        response = model.generate_content(prompt)
        full_text = clean_response(response.text)
        
        # 1. Extraer Sticky Notes
        note_pattern = r"NODE_ID:\s*(.*?)\s*Note_Content:\s*(.*?)\s*End_Note_Content"
        matches = re.findall(note_pattern, full_text, re.DOTALL)
        
        node_map = {n['id']: n for n in nodes}

        for node_id, content in matches:
            target_node = node_map.get(node_id.strip())
            if target_node:
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
                    "id": f"note_{target_node['id']}", 
                    "name": f"Nota: {target_node['name'].split('.')[-1].split('_')[0]}"
                }
                new_notes.append(sticky)

        # 2. Extraer Spec-Sheet
        spec_pattern = r"SPEC_SHEET_START(.*?)SPEC_SHEET_END"
        spec_match = re.search(spec_pattern, full_text, re.DOTALL)
        if spec_match:
            guide_markdown = spec_match.group(1).strip()
        else:
            guide_markdown = "⚠️ No se generó la guía de configuración."

    except Exception as e:
        logger.error(f"Error en Writer: {e}")
        return nodes, f"Error generando guía: {str(e)}"

    return nodes + new_notes, guide_markdown