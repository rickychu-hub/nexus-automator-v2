# app.py (VERSIÓN V4.6 - UI Superpuesta y Arreglada)
import streamlit as st
import requests
import json
import os
import logging
from streamlit_lottie import st_lottie
import time

# --- CONFIGURACIÓN BÁSICA ---
st.set_page_config(page_title="Nexus Automator 🤖", page_icon="🤖", layout="wide")

# URLs (corregidas)
INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")

logger = logging.getLogger(__name__)

# --- ESTILO VISUAL (CON ARREGLO V4.6) ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(180deg, #0a0f13 0%, #111a2f 40%, #0d1317 100%);
        color: #e4e6eb !important;
    }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="stChatMessage-assistant"] { background: rgba(255, 255, 255, 0.04); border-radius: 10px; padding: 12px; margin-bottom: 0.6rem; border-left: 4px solid #00aaff; }
    div[data-testid="stChatMessage-user"] { background: rgba(0, 136, 255, 0.18); border-radius: 10px; padding: 12px; margin-bottom: 0.6rem; border-right: 4px solid #00aaff; }
    div.stMarkdown p, .stMarkdown li, .stTextInput label { color: #f0f2f5 !important; font-size: 16px; }
    button { background-color: #0077ff !important; color: white !important; border: none; border-radius: 8px; font-weight: 600; transition: 0.3s ease; }
    button:hover { background-color: #0099ff !important; box-shadow: 0 0 10px rgba(0,153,255,0.5); }
    input { background-color: #14191e !important; color: #e4e6eb !important; border: 1px solid #00aaff !important; border-radius: 6px !important; }
    .resumen-box { background: rgba(0, 170, 255, 0.08); border-left: 3px solid #00aaff; padding: 12px 18px; border-radius: 8px; margin-bottom: 1rem; color: #cce6ff; }

    </style>
""", unsafe_allow_html=True)

# --- FUNCIÓN PARA CARGAR LOTTIE (V4.1 LOCAL) ---
@st.cache_data
def load_lottie_local(filepath: str):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error cargando Lottie local {filepath}: {e}")
        return None

LOTTIE_FILEPATH = "animacion.json"
lottie_animation = load_lottie_local(LOTTIE_FILEPATH)

# --- TÍTULO ---
st.markdown("<h1 style='text-align:center; color:#00aaff;'>🤖 Nexus Automator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#cccccc;'>Tu Co-Piloto de Automatización con IA y n8n</p>", unsafe_allow_html=True)

# --- ESTADOS DE SESIÓN (Corregidos V4.5) ---
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hola 👋 ¿Qué proceso te gustaría automatizar hoy?"}]
if "conversation_state" not in st.session_state:
    st.session_state.conversation_state = "waiting_for_prompt"
if "interview_history" not in st.session_state:
    st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
if "stored_answers" not in st.session_state:
    st.session_state.stored_answers = {}
if "final_briefing" not in st.session_state:
    st.session_state.final_briefing = ""

# --- FUNCIÓN PARA MOSTRAR MENSAJES ---
def display_message(message):
    with st.chat_message(message["role"]):
        try:
            parsed = json.loads(message["content"])
            st.json(parsed)
        except:
            st.markdown(message["content"])
        if message.get("workflow_json"):
            unique_key = f"download_btn_{hash(message['content'])}"
            st.download_button(
                key=unique_key,
                label="📥 Descargar Workflow.json",
                data=json.dumps(message["workflow_json"], indent=2),
                file_name="workflow.json",
                mime="application/json",
                use_container_width=True
            )

# --- GESTIÓN DE ENTRADA (ARREGLO V2.6) ---
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


# --- LÓGICA DE RENDERIZADO PRINCIPAL ---

# Contenedor para la UI principal (chat y prompt)
main_ui = st.empty()

with main_ui.container():
    # --- MOSTRAR HISTORIAL ---
    for msg in st.session_state.messages:
        display_message(msg)

    # --- PROMPT INICIAL ---
    if st.session_state.conversation_state == "waiting_for_prompt":
        st.info("💡 Describe un proceso (ej: *Cuando se acepte un presupuesto en Zoho, crear factura en Holded y notificar en Trello*).")
        if prompt := st.chat_input("¿Qué automatizamos hoy?"):
            handle_user_input(prompt)

    # --- FORMULARIO DE PREGUNTAS (ARREGLO V2.5) ---
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
                    answers[key] = st.text_input(f"💬 {q}", key=key, value=prev_value, placeholder="Escribe tu respuesta aquí...")

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
                st.session_state.final_briefing = data.get("briefing")
                st.session_state.conversation_state = "generating"
                st.rerun()

            elif data.get("status") == "needs_more_info":
                st.session_state.interview_history["questions"] = data.get("questions", [])
                st.session_state.conversation_state = "waiting_for_answers"
                st.rerun()

        except requests.exceptions.RequestException as e:
            st.error(f"Error de comunicación con el Co-Piloto: {e}")
            st.session_state.conversation_state = "waiting_for_prompt"


# --- GENERACIÓN DEL WORKFLOW (¡ARREGLO V4.6!) ---
# --- GENERACIÓN DEL WORKFLOW FINAL (V4.8 - ARREGLO DE 3 BUGS) ---
if st.session_state.conversation_state == "generating":

    # Ocultamos el chat y el prompt de entrada
    main_ui.empty() 

    # --- Generación del Workflow ---

    # 1. Mostramos el Resumen
    st.markdown("### 🚀 Generando tu Automatización")
    st.markdown(st.session_state.final_briefing)
    st.markdown("---")

    # Banderas para controlar el reinicio
    generation_complete = False

    # El Log de Estado en Vivo
    with st.status("⚙️ El Co-Piloto está trabajando...", expanded=True) as status_ui:

        final_json_str = None
        summary_str = "Resumen no disponible."
        workflow_json_obj = None
        decoded_line = ""

        try:
            # Conectamos al Stream
            response = requests.post(
                GENERATION_URL, 
                json={"user_prompt": st.session_state.final_briefing},
                timeout=600,
                stream=True
            )
            response.raise_for_status()

            # Leemos el Log en Vivo
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')

                    if decoded_line.startswith('{') and decoded_line.endswith('}'):
                        final_json_str = decoded_line
                    elif "ERROR:" in decoded_line:
                        logger.error(f"Error de pipeline V4.9: {decoded_line}")
                        status_ui.update(label=f"❌ Error: {decoded_line}", state="error")
                        time.sleep(5)
                        generation_complete = True
                        break 
                    else:
                        status_ui.write(decoded_line)

            # --- Procesamiento Post-Stream ---
            if final_json_str:
                api_response = json.loads(final_json_str)
                workflow_json_obj = api_response.get("workflow_json")
                summary_str = api_response.get("executive_summary", "Resumen no disponible.")

                status_ui.update(label="✅ ¡Workflow generado con éxito!", state="complete")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"✅ ¡Workflow generado con éxito!\n\n> {summary_str}",
                    "workflow_json": workflow_json_obj
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

    # --- Reinicio después de completar la generación ---
    # El 'rerun' ahora está AFUERA de las columnas,
    # y solo se ejecuta cuando el stream ha terminado.
    if generation_complete:
        # --- Resetear el estado ---
        st.session_state.conversation_state = "waiting_for_prompt"
        st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
        st.session_state.stored_answers = {}
        st.session_state.final_briefing = ""
        time.sleep(1) 
        st.rerun()