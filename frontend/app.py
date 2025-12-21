# frontend/app.py
# VERSIÓN V8.2 - RESTAURACIÓN COMPLETA (IA + MONITORIZACIÓN + LOGIN)

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

# --- 1. CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="Nexus OS", page_icon="⚡", layout="wide")
logger = logging.getLogger(__name__)

# --- HACK DE RUTAS (NIVEL: FUERZA BRUTA) 🦍 ---
# 1. Localizamos dónde estamos (frontend)
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Localizamos la raíz del proyecto (una carpeta atrás)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
# 3. Localizamos explícitamente la carpeta 'backend'
backend_dir = os.path.join(root_dir, 'backend')

# 4. AÑADIMOS TODO AL PATH (Sin pedir permiso)
if root_dir not in sys.path:
    sys.path.append(root_dir)
if backend_dir not in sys.path:
    sys.path.append(backend_dir)

# Configuración básica de logs
logging.basicConfig(level=logging.INFO)

# --- IMPORTACIÓN A PRUEBA DE BOMBAS ---
fetch_and_store_history = None 

try:
    # OPCIÓN A: Ruta directa (gracias a sys.path.append(backend_dir))
    from app.services.n8n_sync import fetch_and_store_history
except ImportError as e1:
    try:
        # OPCIÓN B: Ruta completa estándar
        from backend.app.services.n8n_sync import fetch_and_store_history
    except ImportError as e2:
        # SI TODO FALLA
        error_msg = f"Ruta A: {e1} | Ruta B: {e2}"
        logger.error(f"❌ ERROR CRÍTICO IMPORTANDO BACKEND: {error_msg}")
        def fetch_and_store_history(): 
            return False, f"⚠️ Error: {error_msg}"
# --- 2. AUTENTICACIÓN (Persistente) ---
# Hash para '369852147'
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
        'key': 'nexus_random_signature_key', 
        'name': 'nexus_auth_cookie',
    },
    'pre-authorized': {
        'emails': ['ricardochunas@gmail.com']
    }
}

authenticator = stauth.Authenticate(
    auth_config['credentials'],
    auth_config['cookie']['name'],
    auth_config['cookie']['key'],
    auth_config['cookie']['expiry_days'],
    auth_config['pre-authorized']
)

# --- 3. LOGIN ---
name, authentication_status, username = authenticator.login('main')

if authentication_status:
    # =========================================================
    #  APLICACIÓN PRINCIPAL (Solo visible si estás logueado)
    # =========================================================

    # --- CONFIGURACIÓN DE VARIABLES ---
    INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
    GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")
    REFACTOR_URL = os.getenv("REFACTOR_URL", "http://localhost:8000/refactor-workflow/")
    
    # Supabase (Definido aquí para evitar errores de importación externa)
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")

    @st.cache_resource
    def init_supabase():
        if not SUPABASE_URL or not SUPABASE_KEY:
            return None
        try:
            return create_client(SUPABASE_URL, SUPABASE_KEY)
        except Exception as e:
            logger.error(f"Error Supabase: {e}")
            return None

    supabase = init_supabase()

    # --- ESTADO DE LA SESIÓN (Chatbot) ---
    if "messages" not in st.session_state: st.session_state.messages = []
    if "conversation_state" not in st.session_state: st.session_state.conversation_state = "waiting_for_prompt"
    if "interview_history" not in st.session_state: st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
    if "stored_answers" not in st.session_state: st.session_state.stored_answers = {}
    if "final_briefing" not in st.session_state: st.session_state.final_briefing = ""

    # --- FUNCIONES AUXILIARES ---
    def load_custom_css():
        st.markdown("""
            <style>
            .stApp { background: linear-gradient(180deg, #0e1117 0%, #161b22 100%); color: #c9d1d9; }
            div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
            .deploy-card { background-color: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-top: 10px; }
            .status-badge { background-color: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
            </style>
        """, unsafe_allow_html=True)
    load_custom_css()

    def generating_thoughts_formatter(text: str) -> str:
        return text.replace("Investigator", "🕵️ **Investigator**").replace("Architect", "🏛️ **Architect**").replace("Builder", "🏗️ **Builder**")

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

    def display_message(message):
        with st.chat_message(message["role"]):
            content = message["content"]
            try:
                parsed = json.loads(content)
                st.markdown(parsed.get("executive_summary", content))
            except:
                st.markdown(content)
            
            if message.get("workflow_json"):
                st.download_button("📥 Descargar JSON", json.dumps(message["workflow_json"], indent=2), "workflow.json", "application/json")

    # --- UI PRINCIPAL ---
    
    # Sidebar
    with st.sidebar:
        st.write(f"👤 **{name}**")
        authenticator.logout('Cerrar Sesión', 'sidebar')
        st.divider()
        if st.button("🗑️ Nuevo Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_state = "waiting_for_prompt"
            st.rerun()

    st.title("⚡ Nexus Automator OS")
    
    # PESTAÑAS
    tab_assistant, tab_remix, tab_monitor = st.tabs(["🤖 Asistente", "♻️ Optimizador", "📊 Monitorización"])

    # --- PESTAÑA 1: ASISTENTE ---
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
                    with st.chat_message("assistant"): st.markdown("🤔 Necesito detalles:")
                    with st.form("answers_form"):
                        answers = {}
                        for i, q in enumerate(st.session_state.interview_history["questions"]):
                            key = f"q_{i}"
                            val = st.text_input(f"💬 {q}", key=key)
                            answers[key] = val
                        if st.form_submit_button("Enviar"):
                            st.session_state.stored_answers.update(answers)
                            st.session_state.interview_history["answers"] = list(answers.values())
                            st.session_state.conversation_state = "interviewing"
                            st.rerun()

        # Lógica de entrevista (Backend Call)
        if st.session_state.conversation_state == "interviewing":
            with st.spinner("🧠 Analizando..."):
                try:
                    resp = requests.post(INTERVIEW_URL, json=st.session_state.interview_history, timeout=180)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state.final_briefing = json.dumps(data.get("briefing"), indent=2)
                        st.session_state.conversation_state = "generating"
                        st.rerun()
                    else:
                        st.error("Error en el servidor de entrevista.")
                        st.session_state.conversation_state = "waiting_for_prompt"
                except Exception as e:
                    st.error(f"Error de conexión: {e}")
                    st.session_state.conversation_state = "waiting_for_prompt"

        # Lógica de generación (Backend Call)
        if st.session_state.conversation_state == "generating":
            main_ui.empty()
            with st.status("⚙️ Generando Workflow...", expanded=True) as status:
                log_p = status.empty()
                full_buf = ""
                try:
                    resp = requests.post(GENERATION_URL, json={"user_prompt": st.session_state.final_briefing}, timeout=600, stream=True)
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            full_buf += chunk
                            log_p.markdown(generating_thoughts_formatter(full_buf) + "▌")
                    
                    # Intentar extraer JSON final
                    if "{" in full_buf:
                        final_json_str = full_buf[full_buf.find("{"):]
                        try:
                            api_resp = json.loads(final_json_str)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": f"✅ Hecho.\n> {api_resp.get('executive_summary', '')}",
                                "workflow_json": api_resp.get("workflow_json")
                            })
                            status.update(label="✅ Completado", state="complete")
                        except:
                            status.update(label="⚠️ Error parseando JSON", state="error")
                    
                    st.session_state.conversation_state = "waiting_for_prompt"
                    st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
                    st.rerun()
                except Exception as e:
                    st.error(f"Error generando: {e}")
                    st.session_state.conversation_state = "waiting_for_prompt"

    # --- PESTAÑA 2: OPTIMIZADOR ---
    with tab_remix:
        st.header("♻️ Optimizador")
        uploaded = st.file_uploader("Sube JSON", type=["json"])
        instr = st.text_area("Instrucciones", value=st.session_state.get("remix_prompt", ""))
        if st.button("🚀 Refactorizar") and uploaded and instr:
            st.info("Conectando con el módulo de refactorización...")
            # Aquí iría la llamada a REFACTOR_URL (simplificada para este ejemplo)

    # --- PESTAÑA 3: MONITORIZACIÓN ---
    with tab_monitor:
        st.header("📡 Observabilidad n8n")
        
        col1, _ = st.columns([1, 4])
        with col1:
            if st.button("🔄 Sincronizar Historial n8n"):
                with st.spinner("Sincronizando..."):
                    success, msg = fetch_and_store_history()
                    if success:
                        st.success(msg)
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(msg)
        
        if supabase:
            try:
                res = supabase.table('execution_logs').select("*").order('created_at', desc=True).limit(50).execute()
                if res.data:
                    df = pd.DataFrame(res.data)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No hay datos. Pulsa Sincronizar.")
            except Exception as e:
                st.error(f"Error leyendo logs: {e}")
        else:
            st.warning("No hay conexión con Supabase.")

elif authentication_status is False:
    st.error('Usuario o contraseña incorrectos')
elif authentication_status is None:
    st.warning('Por favor, inicia sesión')