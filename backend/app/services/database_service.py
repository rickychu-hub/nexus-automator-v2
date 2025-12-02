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
def save_workflow_log(prompt: str, workflow_json: dict, summary: str):
    """
    Guarda el resultado de la generación en Supabase.
    Intenta guardar en 'workflows' y en 'generation_logs'.
    """
    db = get_db()
    if not db:
        logger.error("❌ No hay conexión a DB para guardar el workflow.")
        return

    try:
        # 1. Intentar guardar en la tabla principal de workflows
        # NOTA: Ajusta los nombres de las columnas si son diferentes en tu Supabase
        data = {
            "name": summary[:50],  # Usamos el resumen como nombre corto
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
            "created_at": datetime.utcnow().isoformat()
        }
        try:
            db.table("generation_logs").insert(log_data).execute()
            logger.info("📝 Log guardado en 'generation_logs'.")
        except Exception as e:
            logger.warning(f"⚠️ No se pudo guardar en 'generation_logs': {e}")

    except Exception as e:
        logger.error(f"❌ Error general guardando en Supabase: {e}")