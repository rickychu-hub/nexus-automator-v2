# nexus_frontend/app.py
# VERSIÓN V8.0 - AUTH PERSISTENTE & N8N SYNC

import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
import requests
import json
import os
import logging
import time
import re
import datetime
import unicodedata
import pandas as pd
from supabase import create_client, Client
import sys

# --- HACK: IMPORTAR BACKEND ---
# Añadimos el directorio raíz al path para poder importar módulos del backend
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
if root_dir not in sys.path:
    sys.path.append(root_dir)

try:
    from backend.app.services.n8n_sync import fetch_and_store_history
except ImportError:
    # Fallback silencioso si falla la importación para no romper la UI inmediatamente
    def fetch_and_store_history(): return False, "Módulo backend no encontrado"

# --- 1. CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="Nexus OS", page_icon="⚡", layout="wide")

logger = logging.getLogger(__name__)

# --- 2. CONFIGURACIÓN DE AUTH (YAML-like dict) ---
# En producción, esto debería venir de un archivo config.yaml o variables de entorno
hashed_passwords = stauth.Hasher(['369852147', 'demo123']).generate()

auth_config = {
    'credentials': {
        'usernames': {
            'ricardochunas@gmail.com': {
                'name': 'Ricardo Chunas',
                'password': hashed_passwords[0]
            },
            'demo': {
                'name': 'Demo User',
                'password': hashed_passwords[1]
            }
        }
    },
    'cookie': {
        'expiry_days': 30,
        'key': 'random_signature', 
        'name': 'nexus_auth',
    },
    'pre-authorized': {
        'emails': ['ricardochunas@gmail.com']
    }
}

# Inicializar Authenticator
authenticator = stauth.Authenticate(
    auth_config['credentials'],
    auth_config['cookie']['name'],
    auth_config['cookie']['key'],
    auth_config['cookie']['expiry_days'],
    # preassigned_emails=auth_config['pre-authorized']['emails'] # Opcional
)

# --- 3. LOGIC DE LOGIN ---
# Renderizar widget de login
authenticator.login('main')

if st.session_state["authentication_status"]:
    # ---------------------------------------------------------
    # USUARIO AUTENTICADO -> CARGAR APLICACIÓN
    # ---------------------------------------------------------

    # URLs del Backend y Servicios
    INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
    GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")
    REFACTOR_URL = os.getenv("REFACTOR_URL", "http://localhost:8000/refactor-workflow/")
    N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://n8n-motor.onrender.com")

    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

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

    # --- INICIALIZAR SESSION_STATE APP ---
    if "messages" not in st.session_state: st.session_state.messages = []
    if "conversation_state" not in st.session_state: st.session_state.conversation_state = "waiting_for_prompt"
    if "interview_history" not in st.session_state: st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
    if "stored_answers" not in st.session_state: st.session_state.stored_answers = {}
    if "final_briefing" not in st.session_state: st.session_state.final_briefing = ""

    # --- FUNCIONES DE UTILIDAD (CSS & LOGIC) ---
    def load_custom_css():
        st.markdown("""
            <style>
            /* Tema General Dark */
            .stApp { background: linear-gradient(180deg, #0e1117 0%, #161b22 100%); color: #c9d1d9; }
            
            /* Sidebar History Items */
            div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
            section[data-testid="stSidebar"] button {
                white-space: normal !important;
                height: auto !important;
                min-height: 45px !important;
                padding: 8px 12px !important;
                text-align: left !important;
                justify-content: flex-start !important;
                font-size: 12px !important;
                line-height: 1.4 !important;
                border: 1px solid #30363d !important;
                background-color: #161b22 !important;
                color: #c9d1d9 !important;
                transition: all 0.2s ease !important;
            }
            section[data-testid="stSidebar"] button:hover {
                border-color: #8b949e !important;
                background-color: #21262d !important;
            }
            div[data-testid="stChatMessage-assistant"] { background: #161b22; border: 1px solid #30363d; border-radius: 12px; }
            div[data-testid="stChatMessage-user"] { background: #1f6feb20; border: 1px solid #1f6feb; border-radius: 12px; }
            
            .deploy-card {
                background-color: #0d1117;
                border: 1px solid #30363d;
                border-radius: 10px;
                padding: 20px;
                margin-top: 10px;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }
            .deploy-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                border-bottom: 1px solid #30363d;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            .status-badge {
                background-color: #238636;
                color: white;
                padding: 4px 12px;
                border-radius: 20px;
                font-size: 0.8em;
                font-weight: bold;
                text-transform: uppercase;
            }
            .n8n-btn {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background-color: #ff6d5a;
                color: white !important;
                padding: 0.5rem 1rem;
                border-radius: 6px;
                text-decoration: none;
                font-weight: 600;
                width: 100%;
                border: 1px solid #ff6d5a;
                transition: all 0.2s;
            }
            .n8n-btn:hover {
                background-color: #ff8f80;
                border-color: #ff8f80;
            }
            section[data-testid="stSidebar"] {
                min-width: 350px !important;
                width: 400px !important;
            }
            .stCode { font-family: 'Fira Code', monospace !important; }
            .stTabs [data-baseweb="tab-list"] button {
                font-size: 1.2rem;
                padding: 1rem;
            }
            </style>
        """, unsafe_allow_html=True)

    load_custom_css()

    def generating_thoughts_formatter(text: str) -> str:
        text = text.replace("Investigator", "🕵️ **Investigator**")
        text = text.replace("Architect", "🏛️ **Architect**")
        text = text.replace("Builder", "🏗️ **Builder**")
        text = text.replace("Error", "⚠️ **Error**")
        return text

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

    def load_workflow_from_history(record):
        wf_name = record.get("name") or "Workflow Sin Título"
        wf_desc = record.get("description") or "Sin descripción disponible."
        raw_data = record.get("n8n_workflow_id")
        wf_json = {}
        if raw_data:
            if isinstance(raw_data, str):
                try:
                    wf_json = json.loads(raw_data)
                except:
                    st.toast("⚠️ Error JSON", icon="❌")
                    return
            elif isinstance(raw_data, dict):
                wf_json = raw_data
        
        st.session_state.messages = []
        st.session_state.messages.append({
            "role": "assistant",
            "content": json.dumps({
                "executive_summary": f"📂 **{wf_name}**\n\n> {wf_desc}"
            }),
            "workflow_json": wf_json,
            "briefing": wf_desc
        })
        st.session_state.conversation_state = "waiting_for_prompt"

    # --- SIDEBAR ---
    def render_sidebar():
        with st.sidebar:
            st.markdown(f"### 👤 `{st.session_state['name']}`")
            
            # LOGOUT
            authenticator.logout('Cerrar Sesión', 'sidebar')
            
            st.divider()
            st.markdown("### 📂 Historial Reciente")
            
            if supabase:
                try:
                    response = supabase.table("workflows").select("id, name, description, created_at, n8n_workflow_id").order("created_at", desc=True).limit(20).execute()
                    if response.data:
                        for item in response.data:
                            real_name = item.get("name")
                            date_str = item.get("created_at", "")[:10]
                            desc = item.get("description", "")
                            
                            label = real_name if (real_name and len(real_name)>2) else f"📅 {date_str} | {desc[:20]}..."
                            
                            st.button(
                                label,
                                key=item['id'],
                                use_container_width=True,
                                on_click=load_workflow_from_history,
                                args=(item,)
                            )
                    else:
                        st.caption("Vacío.")
                except Exception as e:
                    st.sidebar.error("Error historial.")
            
            st.divider()
            if st.button("🗑️ Nuevo Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_state = "waiting_for_prompt"
                st.rerun()

    def display_message(message):
        with st.chat_message(message["role"]):
            content = message["content"]
            workflow_data = message.get("workflow_json")
            
            try:
                parsed = json.loads(content)
                st.markdown(parsed.get("executive_summary", content))
            except:
                st.markdown(content)

            if workflow_data:
                deployment = workflow_data.get("deployment")
                unique_id = str(time.time_ns())[-6:]
                brief = message.get("briefing", "")
                short_name = generar_nombre_corto(brief)
                file_name = f"{short_name}_{unique_id}.json"
                json_str = json.dumps(workflow_data, indent=2)

                if deployment and deployment.get("status") == "deployed":
                    wf_id = deployment.get('id')
                    dashboard_url = deployment.get("dashboard_url")
                    st.markdown(f"""
                    <div class="deploy-card">
                        <div class="deploy-header">
                            <span class="status-badge">● ACTIVO EN PRODUCCIÓN</span>
                            <span class="id-badge">ID: {wf_id}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if dashboard_url:
                        st.markdown(f'<a href="{dashboard_url}" target="_blank" class="n8n-btn">🌪️ Abrir en n8n</a>', unsafe_allow_html=True)
                
                st.download_button("📥 Descargar JSON", json_str, file_name, "application/json", type="primary")

            guide = message.get("guide", "")
            if guide:
                with st.expander("🛠️ Guía de Conf.", expanded=False):
                    st.markdown(guide)

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

    # --- MAIN APP UI ---
    render_sidebar()

    st.markdown("## 🤖 Nexus Automator")
    st.markdown("Tu Co-Piloto de automatización con **IA + n8n**. Autenticado como: **" + st.session_state["name"] + "**")
    st.markdown("---")

    tab_assistant, tab_remix, tab_monitor = st.tabs(["🤖 Asistente", "♻️ Optimizador", "📊 Monitorización"])

    with tab_assistant:
        main_ui = st.empty()
        with main_ui.container():
            for msg in st.session_state.messages:
                display_message(msg)

            if st.session_state.conversation_state == "waiting_for_prompt":
                st.info("💡 Describe un proceso...")
                if prompt := st.chat_input("¿Qué automatizamos hoy?"):
                    handle_user_input(prompt)

            if st.session_state.conversation_state == "waiting_for_answers":
                if st.session_state.interview_history["questions"]:
                    with st.chat_message("assistant"): st.markdown("🤔 Dudas:")
                    with st.form("answers_form"):
                        answers = {}
                        for i, q in enumerate(st.session_state.interview_history["questions"]):
                            key = f"q_{i}"
                            val = st.text_input(f"💬 {q}", key=key, value=st.session_state.stored_answers.get(key, ""))
                            answers[key] = val
                        if st.form_submit_button("Enviar"):
                            st.session_state.stored_answers.update(answers)
                            st.session_state.interview_history["answers"] = list(answers.values())
                            st.session_state.conversation_state = "interviewing"
                            st.rerun()
                else:
                    st.session_state.conversation_state = "interviewing"
                    st.rerun()

        if st.session_state.conversation_state == "interviewing":
            with st.spinner("🧠 Pensando..."):
                try:
                    resp = requests.post(INTERVIEW_URL, json=st.session_state.interview_history, timeout=180)
                    resp.raise_for_status()
                    data = resp.json()
                    st.session_state.final_briefing = json.dumps(data.get("briefing"), indent=2)
                    st.session_state.conversation_state = "generating"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.session_state.conversation_state = "waiting_for_prompt"

        if st.session_state.conversation_state == "generating":
            main_ui.empty()
            st.markdown("### 🤖 Generando...")
            
            complete = False
            with st.status("⚙️ Trabajando...", expanded=True) as status:
                log_p = status.empty()
                full_buf = ""
                final_json = None
                
                try:
                    resp = requests.post(GENERATION_URL, json={"user_prompt": st.session_state.final_briefing}, timeout=600, stream=True)
                    resp.raise_for_status()
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_buf += chunk
                            log_p.markdown(generating_thoughts_formatter(full_buf) + "▌")
                    
                    # Extraer JSON
                    lines = full_buf.strip().split('\n')
                    pot_json = lines[-1] if lines else ""
                    if pot_json.startswith("{"): final_json = pot_json
                    else:
                        idx = full_buf.rfind('{')
                        if idx != -1: final_json = full_buf[idx:]
                    
                    if final_json:
                        api_resp = json.loads(final_json)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"✅ Hecho.\n> {api_resp.get('executive_summary', '')}",
                            "workflow_json": api_resp.get("workflow_json"),
                            "briefing": st.session_state.final_briefing,
                            "guide": api_resp.get("node_configuration_guide", "")
                        })
                        status.update(label="✅ Listo", state="complete")
                    else:
                        status.update(label="⚠️ Error JSON", state="error")
                    complete = True
                except Exception as e:
                    logger.error(f"Stream error: {e}")
                    status.update(label="❌ Error", state="error")
                    complete = True
            
            if complete:
                st.session_state.conversation_state = "waiting_for_prompt"
                st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
                st.rerun()

    with tab_remix:
        st.header("♻️ Optimizador")
        uploaded = st.file_uploader("Sube JSON", type=["json"])
        instr = st.text_area("Instrucciones", value=st.session_state.get("remix_prompt", ""))
        if st.button("🚀 Refactorizar") and uploaded and instr:
            try:
                raw = json.load(uploaded)
                with st.spinner("Refactorizando..."):
                    res = requests.post(REFACTOR_URL, json={"workflow_json": raw, "instructions": instr}).json()
                    st.success("Refactorizado!")
                    st.download_button("Descargar", json.dumps(res.get("workflow_json"), indent=2), "refactor.json")
            except Exception as e: st.error(str(e))

    with tab_monitor:
        st.header("📡 Observabilidad n8n")
        
        # --- SYNC BUTTON ---
        col_act, col_spacer = st.columns([1, 5])
        with col_act:
            if st.button("🔄 Sincronizar Historial n8n"):
                with st.spinner("Sincronizando con n8n..."):
                    success, msg = fetch_and_store_history()
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)

        if not supabase:
            st.warning("Sin Supabase")
        else:
            try:
                res = supabase.table('execution_logs').select("*").order('created_at', desc=True).limit(200).execute()
                df = pd.DataFrame(res.data)
                if not df.empty:
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No hay logs sincronizados.")
            except Exception as e:
                st.error(f"Error cargando tabla: {e}")

elif st.session_state["authentication_status"] is False:
    st.error('Usuario o contraseña incorrectos')
elif st.session_state["authentication_status"] is None:
    st.warning('Por favor, ingresa tus credenciales')