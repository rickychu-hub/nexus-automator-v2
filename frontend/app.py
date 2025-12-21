# frontend/app.py
# VERSIÓN V8.5 - FULL RESTORE (Sync Local + Optimizador UI)

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
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- 2. CONFIGURACIÓN SUPABASE & N8N (Variables) ---
# Intentamos leer ambas versiones de las variables para ser robustos
INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")
REFACTOR_URL = os.getenv("REFACTOR_URL", "http://localhost:8000/refactor-workflow/")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
N8N_HOST = os.getenv("N8N_HOST") or os.getenv("N8N_BASE_URL")
N8N_API_KEY = os.getenv("N8N_API_KEY")

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

# --- 3. FUNCIÓN SYNC (INTEGRADA AQUÍ) ---
def fetch_and_store_history_local():
    """
    Versión local de la función de sincronización.
    """
    if not N8N_HOST or not N8N_API_KEY:
        # Debug: Mostrar qué falta para ayudar al usuario
        missing = []
        if not N8N_HOST: missing.append("N8N_HOST")
        if not N8N_API_KEY: missing.append("N8N_API_KEY")
        return False, f"❌ Faltan credenciales en Render: {', '.join(missing)}"

    # Endpoint de ejecuciones
    # Aseguramos que el host no termine en barra duplicada
    base_url = N8N_HOST.rstrip('/')
    url = f"{base_url}/api/v1/executions?limit=50&includeData=false"
    headers = {"X-N8N-API-KEY": N8N_API_KEY}

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        executions = data.get("data", [])
        if not executions:
            return True, "No se encontraron ejecuciones nuevas en n8n."

        if not supabase:
            return False, "Error: No hay conexión a Supabase (Revisa SUPABASE_URL/KEY)."

        count = 0
        for exc in executions:
            n8n_id = exc.get("id")
            wf_data = exc.get("workflowData", {})
            wf_name = wf_data.get("name") or "Unknown"
            
            status = exc.get("status")
            if not status:
                if exc.get("finished") is False: status = "running"
                elif exc.get("crashed") is True: status = "crashed"
                else: status = "success"
            
            started_at = exc.get("startedAt")
            stopped_at = exc.get("stoppedAt")
            
            duration_ms = 0
            if started_at and stopped_at:
                try:
                    start_dt = datetime.datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    stop_dt = datetime.datetime.fromisoformat(stopped_at.replace('Z', '+00:00'))
                    duration_ms = int((stop_dt - start_dt).total_seconds() * 1000)
                except: pass

            payload = {
                "n8n_execution_id": n8n_id,
                "workflow_name": wf_name,
                "status": status,
                "created_at": started_at,
                "duration_ms": duration_ms
            }

            try:
                supabase.table("execution_logs").upsert(payload, on_conflict="n8n_execution_id").execute()
                count += 1
            except Exception as e:
                logger.warning(f"Error upserting {n8n_id}: {e}")

        return True, f"✅ Sincronizadas {count} ejecuciones."

    except Exception as e:
        logger.error(f"Error sync n8n: {e}")
        return False, f"Error conexión n8n: {str(e)}"

# --- 4. AUTENTICACIÓN ---
hashed_passwords = stauth.Hasher(['369852147', 'demo123']).generate()

auth_config = {
    'credentials': {
        'usernames': {
            'ricardochunas@gmail.com': {'name': 'Ricardo Chunas', 'password': hashed_passwords[0]},
            'demo': {'name': 'Demo User', 'password': hashed_passwords[1]}
        }
    },
    'cookie': {'expiry_days': 30, 'key': 'nexus_key_v8', 'name': 'nexus_auth'},
    'pre-authorized': {'emails': ['ricardochunas@gmail.com']}
}

authenticator = stauth.Authenticate(
    auth_config['credentials'],
    auth_config['cookie']['name'],
    auth_config['cookie']['key'],
    auth_config['cookie']['expiry_days'],
    auth_config['pre-authorized']
)

name, authentication_status, username = authenticator.login('main')

# --- 5. UI PRINCIPAL ---
if authentication_status:
    
    # Session State
    if "messages" not in st.session_state: st.session_state.messages = []
    if "conversation_state" not in st.session_state: st.session_state.conversation_state = "waiting_for_prompt"
    if "interview_history" not in st.session_state: st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
    if "stored_answers" not in st.session_state: st.session_state.stored_answers = {}
    if "final_briefing" not in st.session_state: st.session_state.final_briefing = ""
    # State para el Optimizador
    if "remix_prompt" not in st.session_state: st.session_state.remix_prompt = ""

    # CSS Global
    st.markdown("""
        <style>
        .stApp { background: linear-gradient(180deg, #0e1117 0%, #161b22 100%); color: #c9d1d9; }
        div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
        .deploy-card { background-color: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-top: 10px; }
        /* Ajuste para el file uploader */
        div[data-testid="stFileUploader"] section { background-color: #161b22; border: 1px dashed #30363d; }
        </style>
    """, unsafe_allow_html=True)

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
    
    tab_assistant, tab_remix, tab_monitor = st.tabs(["🤖 Asistente", "♻️ Optimizador", "📊 Monitorización"])

    # --- TAB MONITORIZACIÓN ---
    with tab_monitor:
        st.header("📡 Observabilidad n8n")
        
        col1, _ = st.columns([1, 4])
        with col1:
            if st.button("🔄 Sincronizar Historial n8n"):
                with st.spinner("Sincronizando..."):
                    success, msg = fetch_and_store_history_local()
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
                    st.dataframe(
                        df, 
                        use_container_width=True,
                        column_config={
                            "status": st.column_config.TextColumn("Estado", width="small"),
                            "created_at": st.column_config.DatetimeColumn("Fecha", format="D MMM, HH:mm"),
                            "duration_ms": st.column_config.NumberColumn("Ms"),
                            "workflow_name": "Workflow"
                        }
                    )
                else:
                    st.info("No hay datos. Pulsa Sincronizar.")
            except Exception as e:
                st.error(f"Error leyendo logs: {e}")
        else:
            st.warning("No hay conexión con Supabase.")

    # --- TAB ASISTENTE ---
    with tab_assistant:
        def handle_user_input(user_input):
            if isinstance(user_input, dict):
                content = "\n".join(f"• {v}" for v in user_input.values())
            else: content = user_input
            st.session_state.messages.append({"role": "user", "content": content})
            if st.session_state.conversation_state == "waiting_for_prompt":
                st.session_state.interview_history["original_prompt"] = content
                st.session_state.final_briefing = content
                st.session_state.conversation_state = "interviewing"
                st.rerun()

        def display_message(msg):
            with st.chat_message(msg["role"]):
                try: st.markdown(json.loads(msg["content"]).get("executive_summary", msg["content"]))
                except: st.markdown(msg["content"])
                if msg.get("workflow_json"):
                    st.download_button("📥 JSON", json.dumps(msg["workflow_json"], indent=2), "wf.json")

        main_ui = st.empty()
        with main_ui.container():
            for msg in st.session_state.messages: display_message(msg)
            
            if st.session_state.conversation_state == "waiting_for_prompt":
                st.info("💡 Describe un proceso...")
                if p := st.chat_input("¿Qué automatizamos?"): handle_user_input(p)

            if st.session_state.conversation_state == "interviewing":
                with st.spinner("🧠 Analizando..."):
                    try:
                        r = requests.post(INTERVIEW_URL, json=st.session_state.interview_history, timeout=120)
                        if r.ok:
                            st.session_state.final_briefing = json.dumps(r.json().get("briefing"), indent=2)
                            st.session_state.conversation_state = "generating"
                            st.rerun()
                    except Exception as e: st.error(f"Error: {e}")

            if st.session_state.conversation_state == "generating":
                st.info("⚙️ Generando...")
                try:
                    r = requests.post(GENERATION_URL, json={"user_prompt": st.session_state.final_briefing}, stream=True, timeout=600)
                    full_resp = ""
                    for chunk in r.iter_content(decode_unicode=True): full_resp += chunk or ""
                    
                    if "{" in full_resp:
                        final_json = json.loads(full_resp[full_resp.find("{"):])
                        st.session_state.messages.append({"role": "assistant", "content": f"✅ Listo.", "workflow_json": final_json.get("workflow_json")})
                    
                    st.session_state.conversation_state = "waiting_for_prompt"
                    st.rerun()
                except Exception as e: st.error(f"Error: {e}")

    # --- TAB OPTIMIZADOR (RESTAURADA) ---
    with tab_remix:
        st.header("♻️ Optimizador")
        st.markdown("Sube un flujo de n8n existente para refactorizarlo o mejorarlo.")
        
        col_up, col_ins = st.columns([1, 1])
        
        with col_up:
            uploaded_file = st.file_uploader("📂 Sube archivo JSON", type=["json"])
        
        with col_ins:
            instructions = st.text_area("📝 Instrucciones de mejora", 
                                      value=st.session_state.remix_prompt,
                                      placeholder="Ej: Añade manejo de errores con Slack...")

        if st.button("🚀 Refactorizar Workflow", type="primary", use_container_width=True):
            if uploaded_file and instructions:
                try:
                    raw_json = json.load(uploaded_file)
                    st.session_state.remix_prompt = instructions # Guardar estado
                    
                    with st.spinner("🔧 El Arquitecto está rediseñando el flujo..."):
                        # Llamada real al backend
                        payload = {"workflow_json": raw_json, "instructions": instructions}
                        response = requests.post(REFACTOR_URL, json=payload, timeout=300)
                        
                        if response.status_code == 200:
                            result = response.json()
                            new_wf = result.get("workflow_json")
                            st.success("✅ ¡Refactorización completada!")
                            st.json(new_wf, expanded=False)
                            st.download_button("📥 Descargar JSON Optimizado", 
                                             json.dumps(new_wf, indent=2), 
                                             "workflow_optimizado.json")
                        else:
                            st.error(f"Error del servidor: {response.text}")
                except Exception as e:
                    st.error(f"Error procesando archivo: {e}")
            else:
                st.warning("⚠️ Por favor sube un archivo JSON y añade instrucciones.")

elif authentication_status is False:
    st.error('Credenciales incorrectas')
elif authentication_status is None:
    st.warning('Por favor, inicia sesión')