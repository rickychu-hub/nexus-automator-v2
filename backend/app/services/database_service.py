import os
from supabase import create_client, Client
import logging
import json
from datetime import datetime

logger = logging.getLogger(__name__)

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")

supabase: Client = None

def init_supabase():
    global supabase
    if url and key:
        try:
            supabase = create_client(url, key)
            logger.info("✅ Cliente Supabase inicializado.")
        except Exception as e:
            logger.error(f"❌ Error inicializando Supabase: {e}")
    else:
        logger.warning("⚠️ Faltan credenciales de Supabase en .env")

def get_db():
    if supabase is None:
        init_supabase()
    return supabase

# --- NUEVA FUNCIÓN DE GUARDADO ---
# --- NUEVA FUNCIÓN DE GUARDADO ---
def save_workflow_log(prompt: str, workflow_json: dict, summary: str, smart_title: str = None, ai_diagnosis: str = None, suggested_fix: str = None):
    """
    Guarda el resultado de la generación en Supabase.
    Intenta guardar en 'workflows' y en 'generation_logs'.
    Ahora soporta smart_title y diagnósticos SRE.
    """
    db = get_db()
    if not db:
        logger.error("❌ No hay conexión a DB para guardar el workflow.")
        return

    try:
        # Inyectar título en metadatos del JSON por si acaso
        if smart_title and isinstance(workflow_json, dict):
            if "meta" not in workflow_json:
                workflow_json["meta"] = {}
            workflow_json["meta"]["smart_title"] = smart_title

        # Priorizar smart_title como nombre, sino fallback al resumen
        final_name = smart_title if smart_title else summary[:50]

        # 1. Intentar guardar en la tabla principal de workflows
        # NOTA: Ajusta los nombres de las columnas si son diferentes en tu Supabase
        data = {
            "name": final_name, 
            "description": prompt,
            "n8n_workflow_id": json.dumps(workflow_json), # Asegúrate que tu columna se llame así o 'json_data'
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Intentamos insertar. Si falla por nombres de columnas, lo registramos.
        try:
            db.table("workflows").insert(data).execute()
            logger.info("💾 Workflow guardado en tabla 'workflows'.")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo guardar en 'workflows' (¿quizás nombres de columna mal?): {e}")

        # 2. Guardar log de auditoría (opcional pero recomendado)
        log_data = {
            "prompt_text": prompt,
            "status": "success",
            "ai_diagnosis": ai_diagnosis,
            "suggested_fix": suggested_fix,
            "created_at": datetime.utcnow().isoformat()
        }
        try:
            db.table("generation_logs").insert(log_data).execute()
            logger.info("📝 Log guardado en 'generation_logs'.")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo guardar en 'generation_logs': {e}")

    except Exception as e:
        logger.error(f"❌ Error general guardando en Supabase: {e}")

# --- REGISTRO DE EJECUCIÓN (SRE Módule) ---
def save_execution_log(workflow_name: str, status: str, error_message: str = None, ai_diagnosis: str = None, suggested_fix: str = None, metadata: dict = None):
    """
    Guarda un log de ejecución de n8n, incluyendo diagnósticos de IA.
    """
    db = get_db()
    if not db: return

    try:
        log_entry = {
            "workflow_name": workflow_name,
            "status": status,
            "error_details": error_message,
            "ai_diagnosis": ai_diagnosis,
            "suggested_fix": suggested_fix,
            "metadata": json.dumps(metadata) if metadata else None,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Guardar en 'execution_logs'
        # Asegúrate de crear esta tabla en Supabase si no existe
        db.table("execution_logs").insert(log_entry).execute()
        logger.info(f"📋 Ejecución registrada: {workflow_name} [{status}]")

    except Exception as e:
        logger.error(f"⚠️ Error guardando execution_log: {e}")