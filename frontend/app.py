# app.py (VERSIÓN V5.0 - UI + Descarga inteligente + Monitorización Supabase)
import streamlit as st
import requests
import json
import os
import logging
import time
import re
import unicodedata
import pandas as pd
from supabase import create_client, Client

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


# --- CONFIGURACIÓN BÁSICA ---
st.set_page_config(page_title="Nexus Automator 🤖", page_icon="🤖", layout="wide")

# URLs del Backend
INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://n8n-motor.onrender.com")

# URLs y Keys de Supabase (Monitorización)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

logger = logging.getLogger(__name__)

# --- INICIALIZAR SUPABASE (CACHÉ) ---
@st.cache_resource
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY:
        return None
    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Error conectando a Supabase: {e}")
        return None

supabase = init_supabase()

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


## --- FUNCIÓN PARA MOSTRAR MENSAJES ---
def display_message(message):
    with st.chat_message(message["role"]):
        # Mostrar contenido normal o JSON
        try:
            parsed = json.loads(message["content"])
            st.json(parsed)
        except:
            st.markdown(message["content"])

        # Si hay workflow generado, activar descarga + botón abrir en n8n
        if message.get("workflow_json"):

            unique_id = str(time.time_ns())[-6:]
            brief = message.get("briefing", "")
            short_name = generar_nombre_corto(brief)
            file_name = f"{short_name}_{unique_id}.json"

            col1, col2 = st.columns([2, 1])

            with col1:
                st.download_button(
                    key=f"download_{unique_id}",
                    label="📥 Descargar Workflow",
                    data=json.dumps(message["workflow_json"], indent=2),
                    file_name=file_name,
                    mime="application/json",
                    use_container_width=True,
                )

            with col2:
                n8n_url = f"{N8N_BASE_URL}/workflow/new"
                st.markdown(
                    f"""
                    <a href="{n8n_url}" target="_blank">
                        <button style="width:100%; margin-top:0.1rem;">
                            🧩 Abrir en n8n
                        </button>
                    </a>
                    """,
                    unsafe_allow_html=True,
                )

            st.caption(
                "ℹ️ Paso rápido: descarga el JSON y, en n8n, usa **Importar > From File** o **From Clipboard**."
            )


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

# --- CABECERA PRINCIPAL ---
st.markdown("## 🤖 Nexus Automator")
st.markdown(
    "Tu Co-Piloto de automatización con **IA + n8n**. <br>"
    "Describe un proceso y generaremos un workflow completo, listo para importar en n8n.",
    unsafe_allow_html=True
)
st.markdown("---")

# --- SISTEMA DE PESTAÑAS (NUEVO V5.0) ---
tab_assistant, tab_monitor = st.tabs(["🤖 Asistente", "📊 Monitorización"])

# ==============================================================================
# PESTAÑA 1: ASISTENTE (Lógica V4.8 Original)
# ==============================================================================
with tab_assistant:
    
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
        
        # Limpiamos solo el contenedor del asistente
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

# ==============================================================================
# PESTAÑA 2: MONITORIZACIÓN (Nueva Lógica con Supabase)
# ==============================================================================
with tab_monitor:
    st.header("Estado del Sistema")
    st.markdown("Visualiza en tiempo real los errores y ejecuciones reportados por n8n.")
    
    if not supabase:
        st.warning("⚠️ Monitorización desactivada: Credenciales de Supabase no encontradas en .env")
    else:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Actualizar Logs", use_container_width=True):
                st.rerun()
        
        try:
            # Consultamos la tabla 'execution_logs' (debe existir en Supabase)
            response = supabase.table('execution_logs').select("*").order('created_at', desc=True).limit(50).execute()
            data = response.data
            
            if data:
                df = pd.DataFrame(data)
                
                # Métricas
                m_col1, m_col2, m_col3 = st.columns(3)
                total_errors = df[df['status'] == 'error'].shape[0]
                total_runs = df.shape[0]
                last_run = df.iloc[0]['created_at']
                
                m_col1.metric("Errores Recientes", total_errors, delta_color="inverse")
                m_col2.metric("Total Logs (Recientes)", total_runs)
                m_col3.metric("Última Actividad", last_run[:19].replace("T", " "))
                
                # Estilizar la tabla
                def color_status(val):
                    color = '#ff4b4b' if val == 'error' else '#00cc99'
                    return f'color: {color}; font-weight: bold'

                st.dataframe(
                    df.style.map(color_status, subset=['status']),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No hay registros de actividad aún en Supabase.")
                
        except Exception as e:
            st.error(f"Error conectando con la Base de Datos: {e}")
            st.caption("Verifica que la tabla 'execution_logs' exista en tu proyecto de Supabase.")