import sentry_sdk
import logging
import os
import google.generativeai as genai
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from datetime import datetime
from pydantic import BaseModel

# Importaciones locales
from app.core.config import settings
from app.services.database_service import init_supabase, save_execution_log
from app.services.chroma_service import init_chroma_client, get_collections, hydrate_knowledge_base
from app.agents.interviewer import agent_interviewer
from app.agents.orchestrator import stream_generation_pipeline
from app.agents.orchestrator import stream_generation_pipeline
from app.agents.debugger import analyze_error
from app.agents.architect import refactor_workflow  # <--- NUEVO IMPORT

# --- CONFIGURACIÓN DE LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURACIÓN DE SENTRY (DETECTOR DE INCENDIOS) ---
# Lo configuramos ANTES de crear la app para atrapar errores de inicio
sentry_dsn = os.getenv("SENTRY_DSN")
if sentry_dsn:
    sentry_sdk.init(
        dsn=sentry_dsn,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )
    logger.info("✅ Sentry iniciado correctamente.")
else:
    logger.warning("⚠️ SENTRY_DSN no encontrado. El monitoreo de errores está desactivado.")

# --- INICIALIZACIÓN DE LA APP (UNA SOLA VEZ) ---
app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# --- MODELOS PYDANTIC (Sugiero mover esto a app/schemas.py en el futuro) ---
class WorkflowRequest(BaseModel):
    user_prompt: str

class InterviewRequest(BaseModel):
    original_prompt: str
    questions: list[str] = []
    answers: list[str] = []

class ErrorReport(BaseModel):
    workflow_name: str
    error_message: str
    workflow_json: dict = None # Contexto opcional para que la IA entienda
    workflow_json: dict = None # Contexto opcional para que la IA entienda
    execution_id: str = None

class RefactorRequest(BaseModel):
    workflow_json: dict
    instructions: str

# --- EVENTOS DE INICIO ---
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Iniciando Nexus Backend (Modular)...")
    
    # 1. Configurar Gemini
    if settings.GOOGLE_API_KEY:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        logger.info("✅ Google Generative AI configurado.")
    else:
        logger.error("❌ ERROR CRÍTICO: GOOGLE_API_KEY no encontrada.")
    
    # 2. Iniciar Memoria (Supabase)
    logger.info("🔌 Iniciando conexión a Memoria (Supabase)...")
    init_supabase()

    # 3. Inicializar conexión a Chroma
    init_chroma_client()

# --- ENDPOINTS ---

@app.get("/")
def read_root():
    return {
        "message": f"{settings.PROJECT_NAME} Operativo",
        "monitoring": "Active" if sentry_dsn else "Inactive",
        "version": settings.VERSION
    }

# Endpoint de prueba para Sentry (Mantenlo oculto o elimínalo en producción real)
@app.get("/sentry-debug")
async def trigger_error():
    logger.info("🔥 Probando Sentry con un error forzado...")
    division_by_zero = 1 / 0

@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/readinessz")
def readinessz():
    kb, exp = get_collections()
    # Verificamos si las colecciones existen (son objetos válidos)
    ready = kb is not None and exp is not None
    return {"ready": ready, "chroma_connected": ready}

@app.post("/system/ingest-kb", tags=["System"])
def ingest_knowledge_base():
    """
    Triggers manual hydration of the Knowledge Base.
    Loads core nodes and massive workflow datasets (ETL + Batching).
    Running synchronously to avoid blocking the event loop (FastAPI spawns a thread).
    """
    logger.info("♻️ Iniciando ingesta manual de Knowledge Base...")
    stats = hydrate_knowledge_base()
    return {"status": "completed", "stats": stats}

@app.post("/interview/")
async def handle_interview(request: InterviewRequest):
    logger.info(f"🎤 Entrevista iniciada para: '{request.original_prompt[:30]}...'")
    try:
        model = genai.GenerativeModel(settings.GENERATIVE_MODEL)
        return agent_interviewer(request.original_prompt, request.questions, request.answers, model)
    except Exception as e:
        logger.error(f"Error en entrevista: {e}", exc_info=True)
        # Sentry capturará esto automáticamente si está configurado
        return {"status": "clarified", "briefing": f"Error: {e}"}

@app.post("/create-workflow-streaming/")
async def handle_create_workflow_streaming(request: WorkflowRequest):
    logger.info(f"⚡ Streaming iniciado para: '{request.user_prompt[:30]}...'")
    return StreamingResponse(
        stream_generation_pipeline(request.user_prompt),
        media_type="text/plain"
    )

@app.post("/webhook/error")
async def handle_n8n_error(report: ErrorReport):
    """
    Endpoint SRE: Recibe errores de n8n, los analiza con IA y guarda el diagnóstico.
    """
    logger.warning(f"🚨 SRE Alert: Error en workflow '{report.workflow_name}'")
    
    # 1. Analizar con IA (Agente Debugger)
    try:
        diagnosis = analyze_error(report.error_message, report.workflow_json)
    except Exception as e:
        logger.error(f"Fallo en Debugger Agent: {e}")
        diagnosis = {"diagnosis": "Error analizando logs", "suggested_fix": "Revisar logs crudos"}

    # 2. Persistir en Supabase (Execution Logs)
    save_execution_log(
        workflow_name=report.workflow_name,
        status="error",
        error_message=report.error_message,
        ai_diagnosis=diagnosis.get("diagnosis"),
        suggested_fix=diagnosis.get("suggested_fix"),
        metadata={"execution_id": report.execution_id}
    )

    return {"status": "recorded", "ai_feedback": diagnosis}

@app.post("/refactor-workflow/")
async def handle_refactor_workflow(request: RefactorRequest):
    """
    Endpoint para Remix & Refactor: Recibe un JSON y instrucciones, devuelve el JSON modificado.
    """
    logger.info(f"♻️ Refactor solicitado: '{request.instructions[:30]}...'")
    try:
        model = genai.GenerativeModel(settings.GENERATIVE_MODEL)
        
        # Llamamos al Arquitecto
        new_json = refactor_workflow(request.workflow_json, request.instructions, model)
        
        # Generar un pequeño resumen del cambio (puedes pedirle a la IA, o hardcodearlo)
        summary = f"Workflow refactorizado según: {request.instructions}"
        
        # Generar Spec-Sheet actualizado (Opcional, reutilizamos el Writer si quisiéramos)
        # Por ahora devolvemos solo el JSON y un resumen simple.
        
        return {
            "status": "success",
            "workflow_json": new_json,
            "executive_summary": f"✅ **Refactor Completado**\n\n> {summary}",
            "node_configuration_guide": "" # Podríamos regenerarla si fuera necesario
        }
    except Exception as e:
        logger.error(f"Error en Refactor: {e}", exc_info=True)
        return {"status": "error", "message": str(e)}