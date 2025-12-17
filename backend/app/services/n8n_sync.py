import os
import requests
import logging
from datetime import datetime
from backend.app.services.database_service import get_db

logger = logging.getLogger(__name__)

def fetch_and_store_history():
    """
    Conecta con la API de n8n, descarga el historial de ejecuciones 
    e inserta/actualiza los registros en Supabase (tabla execution_logs).
    """
    n8n_host = os.getenv("N8N_BASE_URL") or os.getenv("N8N_HOST")
    n8n_api_key = os.getenv("N8N_API_KEY")

    if not n8n_host or not n8n_api_key:
        logger.error("❌ Faltan credenciales N8N_HOST o N8N_API_KEY")
        return False, "Faltan credenciales de n8n"

    # Endpoint de ejecuciones
    # Limitamos a 50 para no saturar, includeData=false para que sea ligero
    url = f"{n8n_host}/api/v1/executions?limit=50&includeData=false"
    headers = {
        "X-N8N-API-KEY": n8n_api_key
    }

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        executions = data.get("data", [])
        if not executions:
            return True, "No se encontraron ejecuciones nuevas."

        db = get_db()
        if not db:
            return False, "No hay conexión a Supabase"

        count = 0
        for exc in executions:
            # Mapeo de campos
            n8n_id = exc.get("id")
            wf_data = exc.get("workflowData", {})
            wf_name = wf_data.get("name") or "Unknown"
            
            # Status: n8n devuelve 'finished', 'crashed', 'error'
            # Pedido explícito: status -> status
            # Intentamos obtener 'status' directo, sino derivamos logicamente
            status = exc.get("status")
            if not status:
                if exc.get("finished") is False:
                    status = "running"
                elif exc.get("crashed") is True:
                    status = "crashed"
                else:
                    status = "success"
            
            started_at = exc.get("startedAt")
            stopped_at = exc.get("stoppedAt")
            
            # Calcular duración
            duration_ms = 0
            if started_at and stopped_at:
                try:
                    start_dt = datetime.fromisoformat(started_at.replace('Z', '+00:00'))
                    stop_dt = datetime.fromisoformat(stopped_at.replace('Z', '+00:00'))
                    duration_ms = int((stop_dt - start_dt).total_seconds() * 1000)
                except:
                    pass

            payload = {
                "n8n_execution_id": n8n_id,
                "workflow_name": wf_name,
                "status": status,
                "created_at": started_at,
                "duration_ms": duration_ms
                # "workflow_json": None # No lo tenemos aquí
            }

            # Upsert basado en n8n_execution_id. 
            try:
                # OJO: Supabase upsert requiere on_conflict
                db.table("execution_logs").upsert(payload, on_conflict="n8n_execution_id").execute()
                count += 1
            except Exception as e:
                logger.warning(f"Error upserting {n8n_id}: {e}")

        return True, f"Sincronizadas {count} ejecuciones."

    except Exception as e:
        logger.error(f"Error fetching n8n history: {e}")
        return False, str(e)
