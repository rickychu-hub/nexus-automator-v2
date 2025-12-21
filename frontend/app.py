# nexus_frontend/app.py
# VERSIÓN V7.1 - SIDEBAR UX (Botones Multilínea + Títulos Reales)

import streamlit as st
import requests
import json
import json
import os
import logging
import time
import re
import datetime # <--- Need this for KPI calculation
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
REFACTOR_URL = os.getenv("REFACTOR_URL", "http://localhost:8000/refactor-workflow/") # <--- NUEVO
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

# --- FUNCIONES DE UTILIDAD (CSS & LOGIC) ---

def load_custom_css():
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
        
        /* Sidebar History Items - ESTILOS MEJORADOS */
        div[data-testid="stSidebar"] { background-color: #0d1117; border-right: 1px solid #30363d; }
        
        /* Inyección específica para botones del sidebar */
        section[data-testid="stSidebar"] button {
            white-space: normal !important;        /* Permitir múltiples líneas */
            height: auto !important;               /* Altura dinámica */
            min-height: 45px !important;           /* Altura mínima para click */
            padding: 8px 12px !important;          /* Padding cómodo */
            text-align: left !important;           /* Alinear texto a la izquierda */
            justify-content: flex-start !important; /* Flex align start */
            font-size: 12px !important;            /* Fuente compacta */
            line-height: 1.4 !important;           /* Espaciado de línea */
            border: 1px solid #30363d !important;  /* Borde sutil */
            background-color: #161b22 !important;  /* Fondo oscuro */
            color: #c9d1d9 !important;             /* Texto claro */
            transition: all 0.2s ease !important;
        }
        
        section[data-testid="stSidebar"] button:hover {
            border-color: #8b949e !important;
            background-color: #21262d !important;
        }

        /* Fix para el texto interior del botón */
        section[data-testid="stSidebar"] button p {
            font-size: 12px !important;
            font-weight: 500 !important;
        }
        
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
        
        /* Botones Personalizados (n8n) */
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
        
        /* Sidebar Width override */
        section[data-testid="stSidebar"] {
            min-width: 350px !important;
            width: 400px !important;
        }
        
        /* Ajustes Streamlit */
        .stCode { font-family: 'Fira Code', monospace !important; }
        
        /* Pestañas más grandes y legibles */
        .stTabs [data-baseweb="tab-list"] button {
            font-size: 1.2rem;
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        </style>
    """, unsafe_allow_html=True)

load_custom_css()

def generating_thoughts_formatter(text: str) -> str:
    """Inyecta emojis en el log de pensamiento para mejor UX."""
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
    """
    Carga un workflow histórico, parsea el JSON y usa el nombre real.
    """
    wf_name = record.get("name") or "Workflow Sin Título"
    wf_desc = record.get("description") or "Sin descripción disponible."
    
    raw_data = record.get("n8n_workflow_id")
    wf_json = {} 
    
    if raw_data:
        if isinstance(raw_data, str):
            try:
                wf_json = json.loads(raw_data)
            except json.JSONDecodeError:
                st.toast("⚠️ Error: El JSON del workflow está corrupto.", icon="❌")
                return
        elif isinstance(raw_data, dict):
            wf_json = raw_data
    else:
        st.toast("⚠️ Registro vacío.", icon="❌")
        return

    # RECONSTRUIR EL CHAT
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
    
    
def render_sidebar():
    with st.sidebar:
        username = st.session_state.get("username", "Usuario")
        st.markdown(f"### 👤 `{username}`")
        
        if st.button("Cerrar Sesión"):
            st.session_state.authenticated = False
            st.rerun()
        
        st.divider()
        st.markdown("### 📂 Historial Reciente")
        
        if supabase:
            try:
                # Obtenemos los últimos 20 workflows
                response = supabase.table("workflows").select("id, name, description, created_at, n8n_workflow_id").order("created_at", desc=True).limit(20).execute()
                
                if response.data:
                    for item in response.data:
                        # 1. Lógica de Título Real
                        # Usamos 'name' como fuente de verdad
                        real_name = item.get("name")
                        date_str = item.get("created_at", "")
                        created_at_fmt = date_str[5:10] if date_str else "??"

                        # Determinar etiqueta del botón
                        if real_name and len(real_name) > 2:
                            # Si hay un nombre válido, lo usamos.
                            # Para UX, podemos poner la fecha pequeña o al inicio
                            # Formato: "Nombre del Proyecto" o "📅 Fecha | Nombre"
                            # El usuario pidió "usa ese título real"
                            button_label = f"{real_name}" 
                        else:
                            # Fallback si no hay título
                            desc = item.get("description", "")
                            fallback_text = (desc[:30] + "...") if desc else "Sin Título"
                            button_label = f"📅 {created_at_fmt} | {fallback_text}"

                        # Renderizar botón con estilo 'multiline' gracias al CSS inyectado
                        st.button(
                            button_label,
                            key=item['id'], 
                            use_container_width=True,
                            on_click=load_workflow_from_history,
                            args=(item,)
                        )
                else:
                    st.caption("No hay historial reciente.")

            except Exception as e:
                st.sidebar.error("Error cargando historial.")
                print(f"DEBUG Error DB: {e}")
                
        else:
            st.warning("DB no conectada")

        st.divider()
        if st.button("🗑️ Nuevo Chat (Limpiar)", use_container_width=True):
            # HARD RESET: Limpieza profunda de toda la sesión
            keys_to_reset = [
                "messages", 
                "conversation_state", 
                "interview_history", 
                "stored_answers", 
                "final_briefing"
            ]
            for key in keys_to_reset:
                if key in st.session_state:
                    del st.session_state[key]
            
            # Re-inicializar básicos para evitar errores antes del rerun
            st.session_state.messages = []
            st.session_state.conversation_state = "waiting_for_prompt"
            st.rerun()

def display_message(message):
    with st.chat_message(message["role"]):
        content = message["content"]
        workflow_data = message.get("workflow_json")
        
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

        if workflow_data:
            deployment = workflow_data.get("deployment")
            unique_id = str(time.time_ns())[-6:]
            brief = message.get("briefing", "")
            short_name = generar_nombre_corto(brief)
            file_name = f"{short_name}_{unique_id}.json"
            json_str = json.dumps(workflow_data, indent=2)

            if deployment and deployment.get("status") == "deployed":
                wf_id = deployment.get('id')
                webhook_url = deployment.get("webhook_url")
                dashboard_url = deployment.get("dashboard_url")
                
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
                        label="📥 Descargar Workflow (.json)",
                        data=json_str,
                        file_name=file_name,
                        mime="application/json",
                        type="primary",
                        use_container_width=True
                    )
            
            else:
                st.download_button(
                    label="📥 Descargar Workflow (.json)",
                    data=json_str,
                    file_name=file_name,
                    mime="application/json",
                    type="primary",
                    use_container_width=True
                )

        # NUEVO: Renderizar Guía de Configuración (Spec-Sheet)
        # Prioridad: configuration_manual > guide
        manual = message.get("configuration_manual") or message.get("guide", "")
        
        if manual:
            with st.expander("🛠️ Guía de Configuración Manual (Spec-Sheet)", expanded=False):
                st.markdown(manual)
                st.info("💡 Copia las expresiones de 'Data Path' tal cual aparecen para evitar errores.")

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
    render_sidebar()

    st.markdown("## 🤖 Nexus Automator")
    st.markdown("Tu Co-Piloto de automatización con **IA + n8n**. Describe un proceso y generaremos un workflow completo.", unsafe_allow_html=True)
    st.markdown("---")

    tab_assistant, tab_remix, tab_monitor = st.tabs(["🤖 Asistente", "♻️ Optimizador / Remix", "📊 Monitorización"])

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

        if st.session_state.conversation_state == "generating":
            main_ui.empty()
            st.markdown("### 🚀 Generando tu Automatización")
            st.markdown(st.session_state.final_briefing)
            st.markdown("---")
            
            complete = False
            with st.status("⚙️ El Co-Piloto está trabajando...", expanded=True) as status:
                log_placeholder = status.empty() # Placeholder para streaming suave
                final_json = None
                summary = ""
                wf_obj = None
                full_response_buffer = "" # Buffer para acumular toda la respuesta
                
                try:
                    # STREAMING SUAVE: iter_content en lugar de iter_lines
                    resp = requests.post(GENERATION_URL, json={"user_prompt": st.session_state.final_briefing}, timeout=600, stream=True)
                    resp.raise_for_status()
                    
                    for chunk in resp.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            # 1. Acumular respuesta completa
                            full_response_buffer += chunk
                            
                            # 2. Actualizar UI en tiempo real con emojis
                            # Solo mostramos el buffer hasta el momento, aplicando formato, 
                            # pero cuidado con re-renderizar todo el texto muy largo.
                            # Opción eficiente: Mostrar las últimas N líneas o todo el buffer procesado.
                            display_text = generating_thoughts_formatter(full_response_buffer)
                            log_placeholder.markdown(display_text + "▌") # Cursor blinking effect
                                
                    
                    # Al finalizar el stream, intentamos extraer el JSON final.
                    # Asumimos que el backend envía el JSON al final o está contenido en la respuesta.
                    # Buscamos el último objeto JSON válido en el buffer si es posible.
                    # O usamos la lógica de líneas si el backend envía líneas limpias.
                    
                    # Intentar parsear la última línea o bloque como JSON
                    lines = full_response_buffer.strip().split('\n')
                    potential_json = lines[-1] if lines else ""
                    
                    if potential_json.startswith("{") and potential_json.endswith("}"):
                        final_json = potential_json
                    else:
                        # Fallback: Intentar encontrar un bloque JSON grande
                        try:
                            # A veces el JSON tiene newlines dentro, buscamos el último '{'
                            last_brace_index = full_response_buffer.rfind('{')
                            if last_brace_index != -1:
                                candidate = full_response_buffer[last_brace_index:]
                                json.loads(candidate) # Validar si es JSON
                                final_json = candidate
                        except:
                            pass

                    if final_json:
                        api_resp = json.loads(final_json)
                        wf_obj = api_resp.get("workflow_json")
                        summary = api_resp.get("executive_summary", "")
                        # Priorizamos el manual personalizado
                        config_manual = api_resp.get("configuration_manual") or api_resp.get("node_configuration_guide", "")

                        # Debug opcional
                        with st.expander("🔍 PAYLOAD RECIBIDO DEL BACKEND (RAW)", expanded=False):
                                st.json(wf_obj)

                        status.update(label="✅ ¡Workflow Generado!", state="complete")
                        
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"✅ ¡Hecho!\n\n> {summary}",
                            "workflow_json": wf_obj,
                            "briefing": st.session_state.final_briefing,
                            "configuration_manual": config_manual 
                        })
                        complete = True
                    else:
                        status.update(label="⚠️ Finalizó pero no se detectó JSON válido.", state="error")
                        logger.error(f"Buffer final: {full_response_buffer}")
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

                st.rerun()

    with tab_remix:
        # Header redundante oculto (ya está en la Tab)
        # st.header("♻️ Optimizador & Refactorización")
        st.markdown("### 🛠️ Centro de Control de Refactorización") # Título más sutil if needed
        st.markdown("Sube un workflow existente (.json) y describe qué cambios quieres hacer. El Arquitecto mantendrá la estructura intacta.")
        
        # --- LÓGICA DE REPARACIÓN (BRIDGE DESDE MONITORIZACIÓN) ---
        pre_filled_instructions = ""
        jump_start_file = None
        
        if "remix_prompt" in st.session_state:
            pre_filled_instructions = st.session_state.pop("remix_prompt")
            st.info(f"🔧 Modo de Reparación Activado: Instrucciones cargadas desde incidente.")
        
        # En el futuro podríamos pre-cargar el archivo si lo tuviéramos en DB
        # if "remix_json" in st.session_state: ...

        uploaded_file = st.file_uploader("Sube tu archivo JSON", type=["json"])
        if uploaded_file:
            st.success("✅ Workflow cargado correctamente.")
            
            # Formulario de instrucciones
            with st.form("refactor_form"):
                instructions = st.text_area(
                    "Instrucciones de cambio (ej: 'Cambia Slack por Discord', 'Añade un filtro de precio > 100')", 
                    value=pre_filled_instructions,
                    height=100
                )
                submitted = st.form_submit_button("🚀 Refactorizar Workflow")
                
                if submitted and instructions:
                    try:
                        # Leer y preparar el JSON
                        raw_json = json.load(uploaded_file)
                        
                        with st.spinner("👷 El Arquitecto está trabajando en los cambios..."):
                            # Llamada al backend
                            payload = {"workflow_json": raw_json, "instructions": instructions}
                            response = requests.post(REFACTOR_URL, json=payload, timeout=60)
                            response.raise_for_status()
                            result = response.json()
                            
                            if result.get("status") == "success":
                                st.balloons()
                                
                                # Simular estructura de mensaje para reutilizar display_message
                                # O simplemente mostrarlo aquí directo
                                output_data = result.get("workflow_json")
                                summary = result.get("executive_summary")
                                
                                st.markdown("### Resultado del Refactor")
                                st.markdown(summary)
                                
                                # Tarjeta de descarga
                                file_name = f"refactor_{int(time.time())}.json"
                                json_str = json.dumps(output_data, indent=2)
                                
                                st.download_button(
                                    label="📥 Descargar Workflow Refactorizado",
                                    data=json_str,
                                    file_name=file_name,
                                    mime="application/json",
                                    type="primary",
                                    use_container_width=True
                                )
                                
                                with st.expander("🔍 Ver JSON Diferencial (Debug)"):
                                    st.json(output_data)
                                    
                            else:
                                st.error(f"Error en backend: {result.get('message')}")
                                
                    except Exception as e:
                        st.error(f"Fallo durante el refactor: {str(e)}")

    with tab_monitor:
        st.header("📡 Dashboard de Observabilidad")
        
        if not supabase:
            st.warning("⚠️ Monitorización desactivada: Credenciales no encontradas.")
        else:
            col_act, col_spacer = st.columns([1, 5])
            with col_act:
                if st.button("🔄 Actualizar Datos"): st.rerun()
            
            try:
                # 1. Fetch de Datos (Con campos detallados)
                res = supabase.table('execution_logs').select(
                    "id, workflow_name, status, created_at, duration_ms, error_message, ai_diagnosis, suggested_fix, workflow_json"
                ).order('created_at', desc=True).limit(200).execute()
                
                if res.data:
                    df = pd.DataFrame(res.data)
                    
                    # Conversión de Tipos
                    if 'created_at' in df.columns:
                        df['created_at'] = pd.to_datetime(df['created_at'])
                    if 'duration_ms' not in df.columns:
                        df['duration_ms'] = 0 # Fallback

                    # PASO 2: KPIs (Top of Page)
                    total_runs = len(df)
                    success_runs = df[df['status'] != 'error'].shape[0]
                    failed_runs = df[df['status'] == 'error'].shape[0]
                    success_rate = (success_runs / total_runs) * 100 if total_runs > 0 else 0
                    
                    # Time Saved: Asumimos 5 min por workflow exitoso
                    time_saved_min = success_runs * 5
                    time_saved_str = f"{time_saved_min} min"
                    if time_saved_min > 60:
                        time_saved_str = f"{time_saved_min / 60:.1f} Horas"

                    # Avg Duration
                    avg_dur_ms = df['duration_ms'].mean()
                    avg_dur_str = f"{avg_dur_ms/1000:.2f}s" if pd.notna(avg_dur_ms) else "N/A"

                    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
                    kpi1.metric("Total Ejecuciones", total_runs)
                    kpi2.metric("Tasa de Éxito", f"{success_rate:.1f}%", delta_color="normal" if success_rate > 90 else "inverse")
                    kpi3.metric("Tiempo Ahorrado ⏳", time_saved_str)
                    kpi4.metric("Duración Promedio", avg_dur_str)
                    
                    st.divider()
                    
                    # PASO 3: TABLA INTERACTIVA
                    st.subheader("📜 Historial de Logs")
                    
                    st.dataframe(
                        df,
                        column_config={
                            "status": st.column_config.TextColumn(
                                "Estado",
                                width="small",
                                help="Estado de la ejecución"
                            ),
                            "workflow_name": st.column_config.TextColumn(
                                "Workflow",
                                width="medium"
                            ),
                            "created_at": st.column_config.DatetimeColumn(
                                "Fecha",
                                format="D MMM, HH:mm",
                                width="small"
                            ),
                            "duration_ms": st.column_config.NumberColumn(
                                "Ms",
                                format="%d ms"
                            ),
                            "ai_diagnosis": None,  # Ocultar
                            "suggested_fix": None, # Ocultar
                            "error_message": None, # Ocultar
                            "workflow_json": None, # Ocultar
                            "id": None
                        },
                        hide_index=True,
                        use_container_width=True
                    )
                    
                    # PASO 4: INSPECTOR DE INCIDENTES
                    st.divider()
                    st.subheader("🔍 Inspector de Incidentes")
                    
                    errors_df = df[df['status'] == 'error']
                    
                    if not errors_df.empty:
                        # Crear opciones legibles
                        options_map = {
                            row['id']: f"⚠️ {row['workflow_name']} ({row['created_at'].strftime('%H:%M')}) - ID: {str(row['id'])[:8]}" 
                            for _, row in errors_df.iterrows()
                        }
                        
                        selected_id = st.selectbox(
                            "Selecciona un workflow fallido para analizar:",
                            options=list(options_map.keys()),
                            format_func=lambda x: options_map[x]
                        )
                        
                        if selected_id:
                            record = errors_df[errors_df['id'] == selected_id].iloc[0]
                            
                            ai_diag = record.get('ai_diagnosis')
                            suggested_fix = record.get('suggested_fix')
                            wf_json_data = record.get('workflow_json')
                            
                            c1, c2, c3 = st.columns(3)
                            
                            with c1: 
                                st.markdown("**🤖 Diagnóstico AI**")
                                if ai_diag:
                                    st.info(ai_diag)
                                else:
                                    st.warning("⚠️ Diagnóstico pendiente. Usa 'Reparar' para analizar.")

                            with c2: 
                                st.markdown("**🔧 Solución Sugerida**")
                                if suggested_fix:
                                    st.success(suggested_fix)
                                else:
                                    st.caption("No disponible.")
                            
                            with c3:
                                st.markdown("**⚙️ Acciones**")
                                if st.button("🛠️ Cargar en Optimizador (Reparar)", key=f"btn_repair_{selected_id}"):
                                    # 1. Cargar Prompt de Reparación
                                    fix_text = suggested_fix if suggested_fix else (ai_diag if ai_diag else record.get('error_message', 'Error desconocido'))
                                    st.session_state['remix_prompt'] = f"Repara este error crítico: {fix_text}"
                                    
                                    # 2. Cargar JSON si existe (FIX REAL)
                                    if wf_json_data:
                                        st.session_state['remix_json'] = wf_json_data # Guardamos en Session
                                        st.toast("✅ Workflow y Error cargados en Optimizador.", icon='🛠️')
                                    else:
                                        st.toast("⚠️ Error cargado, pero falta el JSON del workflow.", icon='⚠️')

                            st.markdown("### 📜 Log Técnico Original (n8n)")
                            st.code(record.get('error_message', 'No logs'), language="text")
                    else:
                        st.success("🎉 No hay incidentes recientes que requieran atención.")

                else:
                    st.info("La base de datos de logs está vacía.")
            except Exception as e:
                st.error(f"Error cargando Dashboard: {e}") 


if __name__ == "__main__":
    if not st.session_state.authenticated:
        login()
    else:
        main_app()