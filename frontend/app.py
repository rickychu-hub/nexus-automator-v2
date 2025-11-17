# app.py (VERSIÓN V4.8 - UI + Descarga inteligente + Fix session_state)
import streamlit as st
import requests
import json
import os
import logging
import time
import re
import unicodedata
import urllib.parse

# --- FUNCIÓN PARA GENERAR NOMBRE CORTO ---
def generar_nombre_corto(briefing_text: str) -> str:
    """
    Genera un nombre corto, limpio y profesional para el workflow
    usando la lógica central del briefing del Agente Entrevistador.
    """
    if not briefing_text:
        return "workflow"

    # 1. pasar a minúsculas
    text = briefing_text.lower()

    # 2. quitar tildes
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

    # 3. palabras clave a detectar
    keywords = [
        "pedidos", "pedido", "leads", "lead", "contactos", "contacto",
        "facturas", "factura", "cliente", "clientes",
        "webhook", "slack", "google", "sheet", "pais", "international"
    ]

    found = [kw for kw in keywords if kw in text]

    if not found:
        return "workflow"

    # 4. dejamos máximo 3 palabras
    name = "_".join(found[:3])

    # 5. limpiamos sobrantes
    name = re.sub(r'[^a-z0-9_]+', '', name)

    return name or "workflow"
def build_n8n_import_url(workflow_json: dict) -> str:
    """
    Crea una URL que abre n8n con el workflow ya cargado en /workflow/new.
    Usa la URL base de N8N_BASE_URL y pasa el JSON en el query param 'data'.
    """
    try:
        raw_json = json.dumps(workflow_json, ensure_ascii=False)
        encoded = urllib.parse.quote(raw_json)
        base = N8N_BASE_URL.rstrip("/")
        # Puedes cambiar /workflow/new por /workflow/import si prefieres otra ruta
        return f"{base}/workflow/new?data={encoded}"
    except Exception as e:
        logger.error(f"Error construyendo URL para n8n: {e}", exc_info=True)
        return ""



# --- CONFIGURACIÓN BÁSICA ---
st.set_page_config(page_title="Nexus Automator 🤖", page_icon="🤖", layout="wide")

# URLs
INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")

N8N_BASE_URL = os.getenv("N8N_BASE_URL", "http://localhost:5678")

logger = logging.getLogger(__name__)

# --- INICIALIZAR SESSION_STATE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "waiting_for_prompt"

if "interview_history" not in st.session_state:
    st.session_state.interview_history = {
        "original_prompt": "",
        "questions": [],
        "answers": []
    }

if "stored_answers" not in st.session_state:
    st.session_state.stored_answers = {}

if "final_briefing" not in st.session_state:
    st.session_state.final_briefing = ""


# --- ESTILO VISUAL ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #0a0f13 0%, #111a2f 40%, #0d1317 100%);
        color: #e4e6eb !important;
    }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="stChatMessage-assistant"] {
        background: rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 0.6rem;
        border-left: 4px solid #00aaff;
    }
    div[data-testid="stChatMessage-user"] {
        background: rgba(0, 136, 255, 0.18);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 0.6rem;
        border-right: 4px solid #00aaff;
    }
    div.stMarkdown p, .stMarkdown li, .stTextInput label {
        color: #f0f2f5 !important;
        font-size: 16px;
    }
    button {
        background-color: #0077ff !important;
        color: white !important;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        transition: 0.3s ease;
    }
    button:hover {
        background-color: #0099ff !important;
        box-shadow: 0 0 10px rgba(0,153,255,0.5);
    }
    input {
        background-color: #14191e !important;
        color: #e4e6eb !important;
        border: 1px solid #00aaff !important;
        border-radius: 6px !important;
    }
    .resumen-box {
        background: rgba(0, 170, 255, 0.08);
        border-left: 3px solid #00aaff;
        padding: 12px 18px;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #cce6ff;
    }
    </style>
""", unsafe_allow_html=True)


def display_message(message):
    with st.chat_message(message["role"]):
        # Mostrar contenido normal o JSON
        try:
            parsed = json.loads(message["content"])
            st.json(parsed)
        except Exception:
            st.markdown(message["content"])

        # Si hay workflow generado, activar descarga + abrir en n8n
        if message.get("workflow_json"):
            workflow_json = message["workflow_json"]

            # ID único seguro para el archivo y botón
            unique_id = str(time.time_ns())[-6:]

            # Nombre corto basado en el briefing
            brief = message.get("briefing", "")
            short_name = generar_nombre_corto(brief)

            # Nombre final del archivo
            file_name = f"{short_name}_{unique_id}.json"

            # Botón de descarga
            st.download_button(
                key=f"download_{unique_id}",
                label="📥 Descargar Workflow",
                data=json.dumps(workflow_json, indent=2),
                file_name=file_name,
                mime="application/json",
                use_container_width=True
            )

            # URL para abrir directamente en n8n
            n8n_url = build_n8n_import_url(workflow_json)
            if n8n_url:
                st.markdown(f"[🔗 Abrir este workflow en n8n]({n8n_url})")

# --- GESTIÓN DE ENTRADA ---
def handle_user_input(user_input):
    if isinstance(user_input, dict):
        answers_text = "\n".join(f"• {v}" for v in user_input.values())
        st.session_state.messages.append({"role": "user", "content": answers_text})
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})

    if st.session_state.conversation_state == "waiting_for_prompt":
        prompt_text = user_input if isinstance(user_input, str) else answers_text
        st.session_state.interview_history["original_prompt"] = prompt_text
        st.session_state.final_briefing = prompt_text
        st.session_state.conversation_state = "interviewing"
        st.rerun()

# --- CABECERA PRINCIPAL (HEADER DE LA APP) ---
st.markdown("## 🤖 Nexus Automator")
st.markdown(
    "Tu Co-Piloto de automatización con **IA + n8n**. <br>"
    "Describe un proceso y generaremos un workflow completo, listo para importar en n8n.",
    unsafe_allow_html=True
)
st.markdown("---")

# --- LÓGICA DE RENDERIZADO PRINCIPAL ---

# Contenedor para la UI principal (chat y prompt)
main_ui = st.empty()


with main_ui.container():
    # Mostrar historial
    for msg in st.session_state.messages:
        display_message(msg)

    # Prompt inicial
    if st.session_state.conversation_state == "waiting_for_prompt":
        st.info("💡 Describe un proceso (ej: *Cuando se acepte un presupuesto en Zoho, crear factura en Holded y notificar en Trello*).")
        if prompt := st.chat_input("¿Qué automatizamos hoy?"):
            handle_user_input(prompt)

    # Formulario de preguntas
    if st.session_state.conversation_state == "waiting_for_answers":
        if st.session_state.interview_history["questions"]:
            with st.chat_message("assistant"):
                if st.session_state.stored_answers:
                    st.markdown("<div class='resumen-box'>", unsafe_allow_html=True)
                    st.markdown("📝 **Tus respuestas anteriores:**", unsafe_allow_html=True)
                    for val in st.session_state.stored_answers.values():
                        st.markdown(f"<p>• {val}</p>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("🤔 Necesito un poco más de información:")

            with st.form("answers_form"):
                answers = {}
                for i, q in enumerate(st.session_state.interview_history["questions"]):
                    key = f"q_{i}"
                    prev_value = st.session_state.stored_answers.get(key, "")
                    answers[key] = st.text_input(
                        f"💬 {q}", key=key, value=prev_value,
                        placeholder="Escribe tu respuesta aquí..."
                    )

                if st.form_submit_button("Enviar respuestas"):
                    for key, value in answers.items():
                        st.session_state.stored_answers[key] = value

                    st.session_state.interview_history["answers"] = list(answers.values())
                    st.session_state.conversation_state = "interviewing"
                    st.rerun()
        else:
            logger.warning("Estado 'waiting_for_answers' sin preguntas. Volviendo a entrevistar.")
            st.session_state.conversation_state = "interviewing"
            st.rerun()


# --- LÓGICA DE ESTADOS (ENTREVISTA) ---
if st.session_state.conversation_state == "interviewing":
    with st.spinner("🧠 El Co-Piloto está pensando..."):
        try:
            response = requests.post(INTERVIEW_URL, json=st.session_state.interview_history, timeout=180)
            response.raise_for_status()
            data = response.json()

            if data.get("status") == "clarified":
                briefing_data = data.get("briefing")

                if isinstance(briefing_data, dict):
                    st.session_state.final_briefing = json.dumps(briefing_data, indent=2, ensure_ascii=False)
                elif isinstance(briefing_data, str):
                    st.session_state.final_briefing = briefing_data
                else:
                    st.session_state.final_briefing = str(briefing_data or "Briefing no disponible.")

                st.session_state.conversation_state = "generating"
                st.rerun()

            elif data.get("status") == "needs_more_info":
                st.session_state.interview_history["questions"] = data.get("questions", [])
                st.session_state.conversation_state = "waiting_for_answers"
                st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"Error de comunicación con el Co-Piloto: {e}")
            st.session_state.conversation_state = "waiting_for_prompt"


# --- GENERACIÓN DEL WORKFLOW FINAL ---
if st.session_state.conversation_state == "generating":

    main_ui.empty()

    st.markdown("### 🚀 Generando tu Automatización")
    st.markdown(st.session_state.final_briefing)
    st.markdown("---")

    generation_complete = False

    with st.status("⚙️ El Co-Piloto está trabajando...", expanded=True) as status_ui:

        final_json_str = None
        summary_str = "Resumen no disponible."
        workflow_json_obj = None
        decoded_line = ""

        try:
            response = requests.post(
                GENERATION_URL,
                json={"user_prompt": st.session_state.final_briefing},
                timeout=600,
                stream=True
            )
            response.raise_for_status()

            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode("utf-8")

                    if decoded_line.startswith("{") and decoded_line.endswith("}"):
                        final_json_str = decoded_line
                    elif "ERROR:" in decoded_line:
                        logger.error(f"Error de pipeline V4.9: {decoded_line}")
                        status_ui.update(label=f"❌ Error: {decoded_line}", state="error")
                        time.sleep(5)
                        generation_complete = True
                        break
                    else:
                        status_ui.write(decoded_line)

            if final_json_str:
                api_response = json.loads(final_json_str)
                workflow_json_obj = api_response.get("workflow_json")
                summary_str = api_response.get("executive_summary", "Resumen no disponible.")

                status_ui.update(label="✅ ¡Workflow generado con éxito!", state="complete")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ ¡Workflow generado con éxito!\n\n> {summary_str}",
                    "workflow_json": workflow_json_obj,
                    "briefing": st.session_state.final_briefing
                })
                generation_complete = True
            else:
                if "ERROR:" not in decoded_line:
                    st.error("❌ Error: El Co-Piloto no devolvió un workflow válido.")
                    status_ui.update(label="❌ Error: No se recibió respuesta final.", state="error")
                    generation_complete = True

        except requests.exceptions.RequestException as e:
            logger.error(f"Error de conexión en V4.9 Stream: {e}", exc_info=True)
            status_ui.update(label=f"❌ Error de conexión: {e}", state="error")
            generation_complete = True

    if generation_complete:
        st.session_state.conversation_state = "waiting_for_prompt"
        st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
        st.session_state.stored_answers = {}
        st.session_state.final_briefing = ""
        time.sleep(1)
        st.rerun()
