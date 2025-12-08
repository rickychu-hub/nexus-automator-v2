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
st.set_page_config(page_title="Nexus Automator 🤖", page_icon="🤖", layout="wide")

# Credenciales de Acceso (HARDCODED PARA MVP - LUEGO MOVER A .ENV)
USERS_DB = {
    "admin": "nexus2025",  # Usuario: Contraseña
    "demo": "demo123"
}

# URLs y Keys
INTERVIEW_URL = os.getenv("INTERVIEW_URL", "http://localhost:8000/interview/")
GENERATION_URL = os.getenv("GENERATION_URL", "http://localhost:8000/create-workflow-streaming/")
N8N_BASE_URL = os.getenv("N8N_BASE_URL", "https://n8n-motor.onrender.com")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

logger = logging.getLogger(__name__)

# Estilos CSS (Incluyendo Login y Sidebar)
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
    }
    
    /* Sidebar History Items */
    .history-item {
        padding: 10px;
        background: #21262d;
        border-radius: 6px;
        margin-bottom: 8px;
        border: 1px solid #30363d;
        cursor: pointer;
        transition: all 0.2s;
    }
    .history-item:hover { border-color: #00aaff; }
    
    /* Mission Control Card (Tu estilo anterior) */
    div[data-testid="stChatMessage-assistant"] { background: #161b22; border: 1px solid #30363d; border-radius: 12px; }
    div[data-testid="stChatMessage-user"] { background: #1f6feb20; border: 1px solid #1f6feb; border-radius: 12px; }
    
    .deploy-card { background-color: #0d1117; border: 1px solid #30363d; border-radius: 10px; padding: 20px; margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
    .deploy-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #30363d; padding-bottom: 10px; margin-bottom: 15px; }
    .status-badge { background-color: #238636; color: white; padding: 4px 12px; border-radius: 20px; font-size: 0.8em; font-weight: bold; }
    .id-badge { font-family: monospace; color: #8b949e; font-size: 0.9em; }
    .n8n-btn { display: inline-flex; align-items: center; justify-content: center; background-color: #ff6d5a; color: white !important; padding: 0.5rem 1rem; border-radius: 6px; text-decoration: none; font-weight: 600; width: 100%; border: 1px solid #ff6d5a; transition: all 0.2s; }
    .n8n-btn:hover { background-color: #ff8f80; border-color: #ff8f80; color: white !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. GESTIÓN DE ESTADO Y SUPABASE ---
@st.cache_resource
def init_supabase():
    if not SUPABASE_URL or not SUPABASE_KEY: return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

supabase = init_supabase()

# Inicialización de Sesión
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if "username" not in st.session_state: st.session_state.username = ""
if "messages" not in st.session_state: st.session_state.messages = []
if "conversation_state" not in st.session_state: st.session_state.conversation_state = "waiting_for_prompt"
if "interview_history" not in st.session_state: st.session_state.interview_history = {"original_prompt": "", "questions": [], "answers": []}
if "stored_answers" not in st.session_state: st.session_state.stored_answers = {}
if "final_briefing" not in st.session_state: st.session_state.final_briefing = ""

# --- 3. FUNCIONES AUXILIARES ---
def generar_nombre_corto(briefing_text: str) -> str:
    # (Tu función existente de nombre corto)
    if not briefing_text: return "workflow"
    text = briefing_text.lower()[:50]
    return re.sub(r'[^a-z0-9]+', '_', text) or "workflow"

def login():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<br><br><br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="login-container">
            <h2>🔐 Nexus Automator v2</h2>
            <p style="color:#8b949e;">Acceso restringido a personal autorizado</p>
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
    """Carga un workflow histórico en el chat"""
    wf_json = record.get("workflow_json", {})
    prompt = record.get("prompt", "Workflow Histórico")
    
    st.session_state.messages = [] # Limpiar chat actual
    st.session_state.messages.append({
        "role": "assistant",
        "content": json.dumps({"executive_summary": f"📂 **Workflow Restaurado:** {prompt}"}),
        "workflow_json": wf_json,
        "briefing": prompt
    })
    st.session_state.conversation_state = "waiting_for_prompt"
    st.rerun()

def render_sidebar():
    with st.sidebar:
        st.write(f"👤 **Usuario:** `{st.session_state.username}`")
        if st.button("Cerrar Sesión"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.divider()
        st.header("📂 Historial")
        
        if supabase:
            # Obtener workflows recientes (Podríamos filtrar por usuario si la DB tuviera user_id)
            try:
                response = supabase.table('workflows').select("*").order('created_at', desc=True).limit(10).execute()
                
                for item in response.data:
                    # Parsear nombre o usar fecha
                    prompt_snippet = item.get("prompt", "Sin título")[:30] + "..."
                    created_at = item.get("created_at", "")[:10]
                    
                    # Botón para cada item del historial
                    if st.button(f"📅 {created_at}\n{prompt_snippet}", key=item['id'], use_container_width=True):
                        load_workflow_from_history(item)
            except Exception as e:
                st.error("Error cargando historial")
        else:
            st.warning("DB no conectada")

        st.divider()
        if st.button("🗑️ Nuevo Chat (Limpiar)", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_state = "waiting_for_prompt"
            st.rerun()

# --- 4. FUNCIÓN DISPLAY MESSAGE (Tu versión PRO) ---
def display_message(message):
    # ... (PEGA AQUÍ TU FUNCIÓN DISPLAY_MESSAGE MEJORADA QUE TE DI EN EL PASO ANTERIOR) ...
    # ... (La que tiene la "Mission Control Card") ...
    # RECUERDA: Copia la función completa del paso anterior para mantener la UI bonita
    with st.chat_message(message["role"]):
        content = message["content"]
        workflow_data = message.get("workflow_json")
        
        if workflow_data:
            try:
                parsed = json.loads(content)
                st.markdown(parsed.get("executive_summary", "✅ Workflow generado."))
            except:
                st.markdown(content)
        else:
            try:
                parsed = json.loads(content)
                if isinstance(parsed, dict) and "executive_summary" not in parsed: st.json(parsed)
                else: st.markdown(content)
            except:
                st.markdown(content)

        if workflow_data:
            deployment = workflow_data.get("deployment")
            json_str = json.dumps(workflow_data, indent=2)
            unique_id = str(time.time())
            
            if deployment and deployment.get("status") == "deployed":
                # UI MISSION CONTROL
                st.markdown(f"""
                <div class="deploy-card">
                    <div class="deploy-header">
                        <span class="status-badge">● ACTIVO EN PRODUCCIÓN</span>
                        <span class="id-badge">ID: {deployment.get('id')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                if deployment.get("webhook_url"):
                    st.caption("🔗 **Webhook:**")
                    st.code(deployment.get("webhook_url"), language="text")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f'<a href="{deployment.get("dashboard_url")}" target="_blank" style="text-decoration:none;"><div class="n8n-btn">🌪️ Abrir en n8n</div></a>', unsafe_allow_html=True)
                with col2:
                    st.download_button("💾 Backup JSON", json_str, file_name=f"wk_{unique_id}.json", mime="application/json", use_container_width=True)
            else:
                 st.download_button("📥 Descargar JSON", json_str, file_name=f"wk_{unique_id}.json", mime="application/json", use_container_width=True)


# --- 5. GESTIÓN DE ENTRADA (Igual que antes) ---
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

# --- 6. MAIN APP FLOW ---
def main_app():
    # Renderizar Sidebar
    render_sidebar()
    
    # Cabecera Principal
    st.title("🤖 Nexus Automator")
    st.caption("Arquitectura Headless v2.0 | Engine: n8n + Gemini Pro")
    
    # Contenedor del Chat
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            display_message(msg)

    # Input Logic (Estados)
    if st.session_state.conversation_state == "waiting_for_prompt":
        if prompt := st.chat_input("¿Qué proceso automatizamos hoy?"):
            handle_user_input(prompt)

    elif st.session_state.conversation_state == "waiting_for_answers":
        # Mostrar formulario de preguntas
        with st.chat_message("assistant"):
            st.write("🕵️ Necesito afinar detalles:")
            with st.form("interview_form"):
                answers = {}
                for i, q in enumerate(st.session_state.interview_history["questions"]):
                    answers[f"q_{i}"] = st.text_input(q)
                if st.form_submit_button("Enviar"):
                    handle_user_input(answers) # Simplificado para el ejemplo

    # ... (AQUÍ IRÍAN TUS BLOQUES DE LOGICA 'interviewing' Y 'generating' QUE YA TIENES) ...
    # Para no hacer el código infinito, COPIA TUS BLOQUES if st.session_state.conversation_state == "interviewing": y "generating": AQUÍ
    
    # BLOQUE DE GENERACIÓN (Resumido para contexto, úsalo completo)
    if st.session_state.conversation_state == "generating":
        with st.status("⚙️ Trabajando en el núcleo...", expanded=True) as status:
            # ... Tu lógica de requests.post al backend ...
            # ... Cuando recibas el final_json ...
            pass # (Mantén tu lógica original aquí)

# --- 7. PUNTO DE ENTRADA ---
if __name__ == "__main__":
    if not st.session_state.authenticated:
        login()
    else:
        main_app()