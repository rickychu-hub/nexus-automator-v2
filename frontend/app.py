# nexus_frontend/app.py
# VERSIÓN V6.0 - FINAL PROD (Auth + Historial + Mission Control UI)

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

# --- 1. CONFIGURACIÓN Y ESTILOS ---
st.set_page_config(page_title="Nexus OS", page_icon="⚡", layout="wide")

# Credenciales de Acceso (Simple para MVP)
USERS_DB = {
    "ricardochunas@gmail.com": "369852147", 
    "demo": "demo123"
}

# URLs del Backend y Servicios
INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://n8n-motor.onrender.com")

# Supabase
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

# --- INICIALIZAR SESSION_STATE ---
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "username" not in st.session_state: st.session_state.username = ""
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_state" not in st.session_state: st.session_state.conversation_state = "waiting_for_prompt"
if "interview_history" not in st.session_state: st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
if "stored_answers" not in st.session_state: st.session_state.stored_answers = {}
if "final_briefing" not in st.session_state: st.session_state.final_briefing = ""

# --- ESTILOS CSS (Mission Control + Login) ---
st.markdown("""
    <style>
    /* Tema General Dark */
    .stApp { background: linear-gradient(180deg, #0e1117 0%, #161b22 100%); color: #c9d1d9; }
    
    /* Login Box */
    .login-container {
        background-color: #161b22;
        padding: 40px;
        border-radius: 10px;
        border: 1px solid #30363d;
        box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Sidebar History Items */
    div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
    
    /* Chat Bubbles */
    div[data-testid="stChatMessage-assistant"] { background: #161b22; border: 1px solid #30363d; border-radius: 12px; }
    div[data-testid="stChatMessage-user"] { background: #1f6feb20; border: 1px solid #1f6feb; border-radius: 12px; }
    
    /* Mission Control Card */
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
        letter-spacing: 0.5px;
    }
    .id-badge {
        font-family: monospace;
        color: #8b949e;
        font-size: 0.9em;
    }
    
    /* Botones Personalizados */
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
        color: white !important;
    }
    
    /* Ajustes Streamlit */
    .stCode { font-family: 'Fira Code', monospace !important; }
    </style>
""", unsafe_allow_html=True)


# --- 2. FUNCIONES DE UTILIDAD ---
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

def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="login-container">
            <h2>🔐 Nexus </h2>
            <p style="color:#8b949e;">Acceso restringido</p>
        </div>
        """, unsafe_allow_html=True)
        
        username = st.text_input("Usuario")
        password = st.text_input("Contraseña", type="password")
        
        if st.button("Iniciar Sesión", use_container_width=True):
            if username in USERS_DB and USERS_DB[username] == password:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("❌ Credenciales incorrectas")

def load_workflow_from_history(record):
    """Carga un workflow histórico usando Callback"""
    
    # 1. Recuperar datos seguros (CORREGIDO)
    # Antes buscabas 'workflow_json', ahora sabemos que está en 'n8n_workflow_id'
    wf_json = record.get("n8n_workflow_id") 
    
    # Antes buscabas 'prompt', ahora usamos 'description' o 'name' como fallback
    prompt_text = record.get("description") or record.get("name") or "Workflow Histórico"
    
    # 2. Protección contra datos corruptos
    if not wf_json:
        st.toast("⚠️ Este registro histórico está vacío o corrupto.", icon="❌")
        return

    # 3. Reconstruir el estado del chat
    st.session_state.messages = [] 
    
    # Creamos un "falso" mensaje del asistente que contiene el workflow
    st.session_state.messages.append({
        "role": "assistant",
        "content": json.dumps({
            "executive_summary": f"📂 **Workflow Restaurado desde el Historial**\n\n> {prompt_text}"
        }),
        "workflow_json": wf_json,
        "briefing": prompt_text
    })
    
    # 4. Forzar estado de espera
    st.session_state.conversation_state = "waiting_for_prompt"
    
    # OPCIONAL: Forzar una recarga suave para ver los cambios inmediatamente si es necesario
    # st.rerun()
def render_sidebar():
    with st.sidebar:
        # Manejo seguro del username
        username = st.session_state.get("username", "Usuario")
        st.markdown(f"### 👤 `{username}`")
        
        if st.button("Cerrar Sesión"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.divider()
        st.markdown("### 📂 Historial Reciente")
        
        if supabase:
            try:
                # 1. CORRECCIÓN: Añadimos 'created_at' y 'description' al select
                # Asumo que 'description' es lo que usabas como 'prompt'.
                response = supabase.table("workflows").select("id, name, description, created_at, n8n_workflow_id").order("created_at", desc=True).limit(10).execute()
                
                # 2. CORRECCIÓN: El bucle va AQUÍ (dentro del try, después del execute), no en el except
                if response.data:
                    for item in response.data:
                        # Usamos 'description' o 'name' porque 'prompt' no existe en tu tabla
                        raw_text = item.get("description") or item.get("name") or "Sin título"
                        
                        # Acortar texto para que quepa en el botón
                        label_short = (raw_text[:28] + "...") if len(raw_text) > 28 else raw_text
                        
                        # Formatear fecha (Manejo seguro si es None)
                        date_str = item.get("created_at", "")
                        created_at_fmt = date_str[5:10] if date_str else "??"

                        # Botón de carga
                        st.button(
                            f"📅 {created_at_fmt} | {label_short}", 
                            key=item['id'], 
                            use_container_width=True,
                            on_click=load_workflow_from_history,
                            args=(item,)
                        )
                else:
                    st.caption("No hay historial reciente.")

            except Exception as e:
                st.sidebar.error("Error cargando historial.")
                print(f"DEBUG Error DB: {e}") # Para que lo veas en tu terminal
                
        else:
            st.warning("DB no conectada")

        st.divider()
        if st.button("🗑️ Nuevo Chat (Limpiar)", use_container_width=True):
            st.session_state.messages = []
            # Asegúrate de reiniciar también el estado del workflow actual
            if 'current_workflow' in st.session_state:
                del st.session_state['current_workflow']
            st.rerun()
def display_message(message):
    with st.chat_message(message["role"]):
        # A. Procesamiento de Texto / JSON
        content = message["content"]
        workflow_data = message.get("workflow_json")
        
        # Intentamos mostrar texto limpio
        if workflow_data:
            try:
                parsed = json.loads(content)
                st.markdown(parsed.get("executive_summary", "✅ Workflow generado con éxito."))
            except:
                st.markdown(content)
        else:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "executive_summary" not in parsed: 
                    st.json(parsed)
                else: 
                    st.markdown(content)
            except:
                st.markdown(content)

        # B. Tarjeta de Despliegue (Mission Control)
        if workflow_data:
            deployment = workflow_data.get("deployment")
            
            # Preparar datos para botones
            unique_id = str(time.time_ns())[-6:]
            brief = message.get("briefing", "")
            short_name = generar_nombre_corto(brief)
            file_name = f"{short_name}_{unique_id}.json"
            json_str = json.dumps(workflow_data, indent=2)

            if deployment and deployment.get("status") == "deployed":
                wf_id = deployment.get('id')
                webhook_url = deployment.get("webhook_url")
                dashboard_url = deployment.get("dashboard_url")
                
                # Renderizar Tarjeta
                st.markdown(f"""
                <div class="deploy-card">
                    <div class="deploy-header">
                        <span class="status-badge">● ACTIVO EN PRODUCCIÓN</span>
                        <span class="id-badge">ID: {wf_id}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if webhook_url:
                    st.caption("🔗 **Webhook Público (Trigger):**")
                    st.code(webhook_url, language="text")
                
                st.write("") 
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    if dashboard_url:
                        st.markdown(f"""
                        <a href="{dashboard_url}" target="_blank" style="text-decoration:none;">
                            <div class="n8n-btn">🌪️ Abrir en n8n</div>
                        </a>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.download_button(
                        label="💾 Descargar Backup (.json)",
                        data=json_str,
                        file_name=file_name,
                        mime="application/json",
                        use_container_width=True
                    )
            
            else:
                # Fallback si no hay deployment automático
                st.warning("⚠️ El workflow fue diseñado, pero la inyección automática no está disponible.")
                st.download_button(
                    label="📥 Descargar JSON para Importación Manual",
                    data=json_str,
                    file_name=file_name,
                    mime="application/json",
                    use_container_width=True
                )

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

# --- 4. APLICACIÓN PRINCIPAL ---
def main_app():
    # Renderizamos la Sidebar
    render_sidebar()

    # Cabecera
    st.markdown("## 🤖 Nexus Automator")
    st.markdown("Tu Co-Piloto de automatización con **IA + n8n**. Describe un proceso y generaremos un workflow completo.", unsafe_allow_html=True)
    st.markdown("---")

    # Pestañas
    tab_assistant, tab_monitor = st.tabs(["🤖 Asistente", "📊 Monitorización"])

    # --- PESTAÑA ASISTENTE ---
    with tab_assistant:
        main_ui = st.empty()
        with main_ui.container():
            # Renderizar mensajes
            for msg in st.session_state.messages:
                display_message(msg)

            # Estado: Esperando Prompt
            if st.session_state.conversation_state == "waiting_for_prompt":
                st.info("💡 Describe un proceso (ej: *Webhook que recibe datos y los manda a Slack*).")
                if prompt := st.chat_input("¿Qué automatizamos hoy?"):
                    handle_user_input(prompt)

            # Estado: Esperando Respuestas (Entrevista)
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

        # Lógica: Entrevistando
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

        # Lógica: Generando
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

                        # Debug opcional
                        with st.expander("🔍 PAYLOAD RECIBIDO DEL BACKEND (RAW)", expanded=False):
                                st.json(wf_obj)

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

    # --- PESTAÑA MONITORIZACIÓN ---
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

# --- 5. PUNTO DE ENTRADA (Flow Auth) ---
if __name__ == "__main__":
    if not st.session_state.authenticated:
        login()
    else:
        main_app()