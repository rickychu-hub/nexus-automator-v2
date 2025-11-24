# backend/app/main.py
import logging
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from datetime import datetime
import google.generativeai as genai

# Importaciones locales (La nueva estructura)
from app.core.config import settings
from app.services.chroma_service import init_chroma_client, get_collections
from app.agents.interviewer import agent_interviewer
from app.agents.orchestrator import stream_generation_pipeline # Pipeline principal

# Configuración de Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# --- Modelos Pydantic (Definidos aquí por simplicidad temporal) ---
from pydantic import BaseModel
class WorkflowRequest(BaseModel): user_prompt: str
class InterviewRequest(BaseModel): original_prompt: str; questions: list[str] = []; answers: list[str] = []

# --- EVENTOS DE INICIO ---
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Iniciando Nexus Backend (Modular)...")
    
    # Configurar Gemini
    if settings.GOOGLE_API_KEY:
        genai.configure(api_key=settings.GOOGLE_API_KEY)
        logger.info("✅ Google Generative AI configurado.")
    else:
        logger.error("❌ ERROR CRÍTICO: GOOGLE_API_KEY no encontrada.")

    # Inicializar conexión a Chroma (Lógica movida a servicio)
    init_chroma_client()

# --- ENDPOINTS ---

@app.post("/interview/")
async def handle_interview(request: InterviewRequest):
    logger.info(f"🎤 Entrevista iniciada para: '{request.original_prompt[:30]}...'")
    try:
        model = genai.GenerativeModel(settings.GENERATIVE_MODEL)
        # Llamamos al agente que ahora vive en su propio archivo
        return agent_interviewer(request.original_prompt, request.questions, request.answers, model)
    except Exception as e:
        logger.error(f"Error en entrevista: {e}", exc_info=True)
        return {"status": "clarified", "briefing": f"Error: {e}"}

@app.post("/create-workflow-streaming/")
async def handle_create_workflow_streaming(request: WorkflowRequest):
    logger.info(f"⚡ Streaming iniciado para: '{request.user_prompt[:30]}...'")
    return StreamingResponse(
        stream_generation_pipeline(request.user_prompt),
        media_type="text/plain"
    )

@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

# Mantenemos endpoints de debug si los necesitas, pero limpios
@app.get("/readinessz")
def readinessz():
    kb, exp = get_collections()
    ready = kb is not None and exp is not None
    return {"ready": ready, "chroma_connected": ready}