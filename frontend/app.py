# app.py (VERSIÓN V5.1 - UI Headless + Monitorización + Clean Code)
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
    if not briefing_text: return "workflow"
    text = briefing_text.lower()
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    keywords = ["pedidos", "leads", "facturas", "cliente", "webhook", "slack", "google", "sheet"]
    found = [kw for kw in keywords if kw in text]
    if not found: return "workflow"
    name = "_".join(found[:3])
    name = re.sub(r'[^a-z0-9_]+', '', name)
    return name or "workflow"

# --- CONFIGURACIÓN BÁSICA ---
st.set_page_config(page_title="Nexus Automator 🤖", page_icon="🤖", layout="wide")

# URLs del Backend
INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://n8n-motor.onrender.com")

# URLs y Keys de Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

logger = logging.getLogger(__name__)

# --- INICIALIZAR SUPABASE ---
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

# --- INICIALIZAR SESSION_STATE (Optimizada) ---
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_state" not in st.session_state: st.session_state.conversation_state = "waiting_for_prompt"
if "interview_history" not in st.session_state: st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
if "stored_answers" not in st.session_state: st.session_state.stored_answers = {}
if "final_briefing" not in st.session_state: st.session_state.final_briefing = ""

# --- ESTILO VISUAL ---
st.markdown("""
    <style>
    .stApp { background: linear-gradient(180deg, #0a0f13 0%, #111a2f 40%, #0d1317 100%); color: #e4e6eb !important; }
    .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    div[data-testid="stChatMessage-assistant"] { background: rgba(255, 255, 255, 0.04); border-radius: 10px; padding: 12px; border-left: 4px solid #00aaff; }
    div[data-testid="stChatMessage-user"] { background: rgba(0, 136, 255, 0.18); border-radius: 10px; padding: 12px; border-right: 4px solid #00aaff; }
    button { background-color: #0077ff !important; color: white !important; border-radius: 8px; font-weight: 600; }
    .resumen-box { background: rgba(0, 170, 255, 0.08); border-left: 3px solid #00aaff; padding: 12px 18px; border-radius: 8px; margin-bottom: 1rem; color: #cce6ff; }
    .success-box { background: rgba(0, 255, 128, 0.1); border: 1px solid #00ff80; padding: 15px; border-radius: 8px; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)


## --- FUNCIÓN PARA MOSTRAR MENSAJES (LÓGICA HEADLESS) ---
def display_message(message):
    with st.chat_message(message["role"]):
        try:
            parsed = json.loads(message["content"])
            st.json(parsed)
        except:
            st.markdown(message["content"])

        # Si hay workflow generado, decidimos qué mostrar
        workflow_data = message.get("workflow_json")
        if workflow_data:
            
            # A. VERIFICAR DESPLIEGUE AUTOMÁTICO (HEADLESS)
            deployment = workflow_data.get("deployment")
            
            if deployment and deployment.get("status") == "deployed":
                wf_id = deployment.get("id")
                dashboard_url = deployment.get("dashboard_url")
                webhook_url = deployment.get("webhook_url")
                
                st.markdown(f"""
                <div class="success-box">
                    <h4>🚀 ¡Workflow Inyectado en n8n!</h4>
                    <p>El sistema ha creado y activado el workflow automáticamente.</p>
                </div>
                """, unsafe_allow_html=True)
                
                if webhook_url:
                    st.markdown("**🔗 Tu Webhook Público:**")
                    st.code(webhook_url, language="text")
                    st.caption("Copia esta URL y úsala en tu Trigger externo.")
                
                if dashboard_url:
                    st.markdown(
                        f"""
                        <a href="{dashboard_url}" target="_blank">
                            <button style="width:100%; margin-top:10px; padding: 10px; cursor: pointer;">
                                🛠️ Ver Workflow en Vivo (ID: {wf_id})
                            </button>
                        </a>
                        """,
                        unsafe_allow_html=True
                    )
            
            else:
                # B. FALLBACK MANUAL (Descarga clásica)
                unique_id = str(time.time_ns())[-6:]
                brief = message.get("briefing", "")
                short_name = generar_nombre_corto(brief)
                file_name = f"{short_name}_{unique_id}.json"

                col1, col2 = st.columns([2, 1])
                with col1:
                    st.download_button(
                        key=f"dl_{unique_id}",
                        label="📥 Descargar JSON",
                        data=json.dumps(workflow_data, indent=2),
                        file_name=file_name,
                        mime="application/json",
                        use_container_width=True,
                    )
                with col2:
                    n8n_url = f"{N8N_BASE_URL}/workflow/new"
                    st.markdown(
                        f"""<a href="{n8n_url}" target="_blank"><button style="width:100%;">🧩 Abrir n8n</button></a>""",
                        unsafe_allow_html=True,
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

# --- CABECERA ---
st.markdown("## 🤖 Nexus Automator")
st.markdown("Tu Co-Piloto de automatización con **IA + n8n**. Describe un proceso y generaremos un workflow completo.", unsafe_allow_html=True)
st.markdown("---")

# --- PESTAÑAS ---
tab_assistant, tab_monitor = st.tabs(["🤖 Asistente", "📊 Monitorización"])

# ==============================================================================
# PESTAÑA 1: ASISTENTE
# ==============================================================================
with tab_assistant:
    main_ui = st.empty()
    with main_ui.container():
        for msg in st.session_state.messages:
            display_message(msg)

        if st.session_state.conversation_state == "waiting_for_prompt":
            st.info("💡 Describe un proceso (ej: *Webhook que recibe datos y los manda a Slack*).")
            if prompt := st.chat_input("¿Qué automatizamos hoy?"):
                handle_user_input(prompt)

        if st.session_state.conversation_state == "waiting_for_answers":
            if st.session_state.interview_history["questions"]:
                with st.chat_message("assistant"):
                    st.markdown("🤔 Necesito un poco más de información:")
                with st.form("answers_form"):
                    answers = {}
                    for i, q in enumerate(st.session_state.interview_history["questions"]):
                        key = f"q_{i}"
                        prev_val = st.session_state.stored_answers.get(key, "")
                        answers[key] = st.text_input(f"💬 {q}", key=key, value=prev_val)
                    if st.form_submit_button("Enviar respuestas"):
                        for k, v in answers.items(): st.session_state.stored_answers[k] = v
                        st.session_state.interview_history["answers"] = list(answers.values())
                        st.session_state.conversation_state = "interviewing"
                        st.rerun()
            else:
                st.session_state.conversation_state = "interviewing"
                st.rerun()

    # --- LÓGICA DE ENTREVISTA ---
    if st.session_state.conversation_state == "interviewing":
        with st.spinner("🧠 El Co-Piloto está pensando..."):
            try:
                resp = requests.post(INTERVIEW_URL, json=st.session_state.interview_history, timeout=180)
                resp.raise_for_status()
                data = resp.json()
                if data.get("status") == "clarified":
                    briefing = data.get("briefing")
                    st.session_state.final_briefing = json.dumps(briefing, indent=2) if isinstance(briefing, dict) else str(briefing)
                    st.session_state.conversation_state = "generating"
                    st.rerun()
                elif data.get("status") == "needs_more_info":
                    st.session_state.interview_history["questions"] = data.get("questions", [])
                    st.session_state.conversation_state = "waiting_for_answers"
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.session_state.conversation_state = "waiting_for_prompt"

    # --- GENERACIÓN DEL WORKFLOW ---
    if st.session_state.conversation_state == "generating":
        main_ui.empty()
        st.markdown("### 🚀 Generando tu Automatización")
        st.markdown(st.session_state.final_briefing)
        st.markdown("---")
        
        complete = False
        with st.status("⚙️ El Co-Piloto está trabajando...", expanded=True) as status:
            final_json = None
            summary = ""
            wf_obj = None
            try:
                resp = requests.post(GENERATION_URL, json={"user_prompt": st.session_state.final_briefing}, timeout=600, stream=True)
                resp.raise_for_status()
                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("{") and decoded.endswith("}"):
                            final_json = decoded
                        elif "ERROR:" in decoded:
                            status.update(label=f"❌ {decoded}", state="error")
                            complete = True
                            break
                        else:
                            status.write(decoded)
                
                if final_json:
                    api_resp = json.loads(final_json)
                    wf_obj = api_resp.get("workflow_json")
                    summary = api_resp.get("executive_summary", "")

                    with st.expander("🔍 PAYLOAD RECIBIDO DEL BACKEND (RAW)", expanded=True):
                            st.json(wf_obj)
                            if "deployment" not in wf_obj:
                                st.error("❌ ALERTA: El objeto 'deployment' NO vino del backend.")
                            else:
                                st.success(f"✅ Deployment Data: {wf_obj['deployment']}")

                    status.update(label="✅ ¡Workflow Generado!", state="complete")
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"✅ ¡Hecho!\n\n> {summary}",
                        "workflow_json": wf_obj,
                        "briefing": st.session_state.final_briefing
                    })
                    complete = True
            except Exception as e:
                logger.error(f"Error stream: {e}")
                status.update(label="❌ Error de conexión", state="error")
                complete = True

        if complete:
            st.session_state.conversation_state = "waiting_for_prompt"
            st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
            st.session_state.stored_answers = {}
            st.session_state.final_briefing = ""
            time.sleep(1)
            st.rerun()

# ==============================================================================
# PESTAÑA 2: MONITORIZACIÓN
# ==============================================================================
with tab_monitor:
    st.header("Estado del Sistema")
    if not supabase:
        st.warning("⚠️ Monitorización desactivada: Credenciales no encontradas.")
    else:
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Actualizar Logs", use_container_width=True): st.rerun()
        
        try:
            res = supabase.table('execution_logs').select("*").order('created_at', desc=True).limit(50).execute()
            if res.data:
                df = pd.DataFrame(res.data)
                col1, col2 = st.columns(2)
                col1.metric("Errores", df[df['status'] == 'error'].shape[0])
                col2.metric("Logs", df.shape[0])
                
                def color_status(val): return f'color: {"#ff4b4b" if val == "error" else "#00cc99"}; font-weight: bold'
                st.dataframe(df.style.map(color_status, subset=['status']), use_container_width=True, hide_index=True)
            else:
                st.info("Sin actividad reciente.")
        except Exception as e:
            st.error(f"Error DB: {e}")