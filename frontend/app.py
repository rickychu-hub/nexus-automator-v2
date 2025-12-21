# frontend/app.py
# VERSIÓN V8.6 - SMART NAME MAPPING & ERROR DEEP SCAN

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

# --- 1. CONFIGURACIÓN ---
st.set_page_config(page_title="Nexus OS", page_icon="⚡", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")
REFACTOR_URL = os.getenv("REFACTOR_URL", "http://localhost:8000/refactor-workflow/")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
N8N_HOST = os.getenv("N8N_HOST") or os.getenv("N8N_BASE_URL")
N8N_API_KEY = os.getenv("N8N_API_KEY")

@st.cache_resource
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY: return None
    try: return create_client(SUPABASE_URL, SUPABASE_KEY)
    except: return None

supabase = init_supabase()

# --- 2. FUNCIÓN DE SINCRONIZACIÓN INTELIGENTE ---
def fetch_and_store_history_local():
    if not N8N_HOST or not N8N_API_KEY:
        return False, "❌ Faltan credenciales N8N"

    base_url = N8N_HOST.rstrip('/')
    headers = {"X-N8N-API-KEY": N8N_API_KEY}

    try:
        # PASO A: OBTENER MAPA DE NOMBRES (ID -> NOMBRE)
        # Esto soluciona el problema de "Unknown"
        wf_map = {}
        try:
            wf_resp = requests.get(f"{base_url}/api/v1/workflows", headers=headers, timeout=5)
            if wf_resp.ok:
                for wf in wf_resp.json().get("data", []):
                    wf_map[wf["id"]] = wf["name"]
        except Exception as e:
            logger.warning(f"No se pudieron cargar nombres de workflows: {e}")

        # PASO B: OBTENER EJECUCIONES (Con detalles)
        url = f"{base_url}/api/v1/executions?limit=30&includeData=true"
        response = requests.get(url, headers=headers, timeout=20)
        response.raise_for_status()
        data = response.json()
        executions = data.get("data", [])

        if not executions: return True, "No hay ejecuciones recientes."
        if not supabase: return False, "Sin Supabase."

        count = 0
        for exc in executions:
            n8n_id = exc.get("id")
            wf_id = exc.get("workflowId")
            
            # 1. RESOLVER NOMBRE (Usando el mapa o los datos del log)
            wf_name = wf_map.get(wf_id, "Unknown Workflow")
            if wf_name == "Unknown Workflow":
                wf_name = exc.get("workflowData", {}).get("name", "Unknown")

            # 2. DETERMINAR ESTADO
            status = "unknown"
            if exc.get("finished") is False: status = "running"
            elif exc.get("crashed") is True: status = "crashed"
            elif exc.get("data", {}).get("resultData", {}).get("error"): status = "error"
            else: status = "success"

            # 3. EXTRAER MENSAJE DE ERROR REAL
            error_msg = None
            if status in ["error", "crashed"]:
                try:
                    res_data = exc.get("data", {}).get("resultData", {})
                    # Intento 1: Error global
                    if res_data.get("error"):
                        error_msg = res_data["error"].get("message") or str(res_data["error"])
                    # Intento 2: Buscar nodo por nodo
                    if not error_msg and res_data.get("runData"):
                        for node_runs in res_data["runData"].values():
                            for run in node_runs:
                                if run.get("error"):
                                    error_msg = run["error"].get("message")
                                    break
                except: error_msg = "Error parsing log"

            # 4. FECHAS
            started_at = exc.get("startedAt")
            stopped_at = exc.get("stoppedAt")
            duration_ms = 0
            if started_at and stopped_at:
                try:
                    s = datetime.datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    e = datetime.datetime.fromisoformat(stopped_at.replace('Z', '+00:00'))
                    duration_ms = int((e - s).total_seconds() * 1000)
                except: pass

            payload = {
                "n8n_execution_id": n8n_id,
                "workflow_name": wf_name, # ¡Ahora sí tendrá nombre!
                "status": status,
                "created_at": started_at,
                "duration_ms": duration_ms,
                "error_message": error_msg
            }

            try:
                supabase.table("execution_logs").upsert(payload, on_conflict="n8n_execution_id").execute()
                count += 1
            except Exception as e: logger.warning(f"Upsert error: {e}")

        return True, f"✅ Sincronizados {count} registros (Limpios)."

    except Exception as e:
        return False, f"Error: {str(e)}"

# --- 3. AUTH & UI (Igual que antes) ---
hashed_passwords = stauth.Hasher(['369852147', 'demo123']).generate()
auth_config = {
    'credentials': {'usernames': {'ricardochunas@gmail.com': {'name': 'Ricardo Chunas', 'password': hashed_passwords[0]}, 'demo': {'name': 'Demo User', 'password': hashed_passwords[1]}}},
    'cookie': {'expiry_days': 30, 'key': 'nexus_key_v8', 'name': 'nexus_auth'},
    'pre-authorized': {'emails': ['ricardochunas@gmail.com']}
}
authenticator = stauth.Authenticate(auth_config['credentials'], auth_config['cookie']['name'], auth_config['cookie']['key'], auth_config['cookie']['expiry_days'], auth_config['pre-authorized'])
name, authentication_status, username = authenticator.login('main')

if authentication_status:
    if "messages" not in st.session_state: st.session_state.messages = []
    if "conversation_state" not in st.session_state: st.session_state.conversation_state = "waiting_for_prompt"
    if "interview_history" not in st.session_state: st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
    if "remix_prompt" not in st.session_state: st.session_state.remix_prompt = ""
    if "final_briefing" not in st.session_state: st.session_state.final_briefing = ""

    st.markdown("""<style>.stApp { background: linear-gradient(180deg, #0e1117 0%, #161b22 100%); color: #c9d1d9; }
        div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
        .deploy-card { background-color: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-top: 10px; }
        div[data-testid="stFileUploader"] section { background-color: #161b22; border: 1px dashed #30363d; }
    </style>""", unsafe_allow_html=True)

    with st.sidebar:
        st.write(f"👤 **{name}**")
        authenticator.logout('Cerrar Sesión', 'sidebar')
        st.divider()
        st.subheader("📂 Biblioteca")
        if supabase:
            try:
                # Consultamos los últimos 5 workflows guardados
                response = supabase.table('workflows').select("name, created_at").order('created_at', desc=True).limit(5).execute()
                
                if response.data:
                    for wf in response.data:
                        wf_name = wf.get('name', 'Sin Nombre')
                        # Mostramos nombre y fecha
                        st.markdown(f"📄 **{wf_name}**")
                        st.caption(f"📅 {wf.get('created_at', '')[:10]}")
                else:
                    st.caption("📭 No hay workflows guardados en la DB.")
            except Exception:
                st.caption("⚠️ Error cargando historial.")
        if st.button("🗑️ Nuevo Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    st.title("⚡ Nexus Automator OS")
    tab_assistant, tab_remix, tab_monitor = st.tabs(["🤖 Asistente", "♻️ Optimizador", "📊 Monitorización"])

    with tab_monitor:
        st.header("📡 Observabilidad n8n")
        col1, _ = st.columns([1, 4])
        with col1:
            if st.button("🔄 Sincronizar Historial n8n"):
                with st.spinner("Analizando historial..."):
                    success, msg = fetch_and_store_history_local()
                    if success: st.success(msg); time.sleep(1); st.rerun()
                    else: st.error(msg)
        
        if supabase:
            try:
                # Mostrar solo lo importante: Workflow, Estado y MENSAJE DE ERROR
                res = supabase.table('execution_logs').select("*").order('created_at', desc=True).limit(50).execute()
                if res.data:
                    df = pd.DataFrame(res.data)
                    st.dataframe(df, use_container_width=True, column_config={
                        "status": st.column_config.TextColumn("Estado"),
                        "workflow_name": st.column_config.TextColumn("Workflow", width="medium"),
                        "error_message": st.column_config.TextColumn("Causa del Fallo 🛑", width="large"),
                        "created_at": st.column_config.DatetimeColumn("Fecha", format="D MMM, HH:mm"),
                        "duration_ms": st.column_config.NumberColumn("Ms")
                    })
                else: st.info("Historial limpio.")
            except Exception as e: st.error(f"Error tabla: {e}")

    with tab_assistant:
        def handle_user_input(user_input):
            content = "\n".join(f"• {v}" for v in user_input.values()) if isinstance(user_input, dict) else user_input
            st.session_state.messages.append({"role": "user", "content": content})
            if st.session_state.conversation_state == "waiting_for_prompt":
                st.session_state.final_briefing = content
                st.session_state.conversation_state = "interviewing"
                st.rerun()

        def display_message(msg):
            with st.chat_message(msg["role"]):
                try: st.markdown(json.loads(msg["content"]).get("executive_summary", msg["content"]))
                except: st.markdown(msg["content"])
                if msg.get("workflow_json"): st.download_button("📥 JSON", json.dumps(msg["workflow_json"], indent=2), "wf.json")

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

    with tab_remix:
        st.header("♻️ Optimizador")
        col_up, col_ins = st.columns([1, 1])
        with col_up: uploaded_file = st.file_uploader("📂 Sube archivo JSON", type=["json"])
        with col_ins: instructions = st.text_area("📝 Instrucciones", value=st.session_state.remix_prompt)
        if st.button("🚀 Refactorizar", type="primary"):
            if uploaded_file and instructions:
                try:
                    raw_json = json.load(uploaded_file)
                    st.session_state.remix_prompt = instructions
                    with st.spinner("Trabajando..."):
                        res = requests.post(REFACTOR_URL, json={"workflow_json": raw_json, "instructions": instructions}, timeout=300)
                        if res.ok:
                            new_wf = res.json().get("workflow_json")
                            st.success("✅ Hecho!")
                            st.download_button("📥 Descargar", json.dumps(new_wf, indent=2), "optimizado.json")
                except Exception as e: st.error(f"Error: {e}")

elif authentication_status is False: st.error('Incorrecto')
elif authentication_status is None: st.warning('Login necesario')