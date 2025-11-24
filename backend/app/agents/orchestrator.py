# backend/app/agents/orchestrator.py
import logging
import json
import asyncio
import google.generativeai as genai
from app.core.config import settings

# Importamos los agentes y servicios
from app.services.chroma_service import get_kb_memory
from app.agents.investigator import agent_investigator
from app.agents.architect import agent_architect
from app.agents.builder import build_nodes_from_plan, final_assembler
from app.agents.writer import agent_technical_writer

logger = logging.getLogger(__name__)

async def stream_generation_pipeline(user_prompt: str):
    """
    Pipeline V4.0 (Async/Stream):
    Investigador -> Arquitecto -> Builder -> Writer -> Assembler
    """
    logger.info(f"⚡ Pipeline iniciado para: {user_prompt[:50]}...")
    
    try:
        # Inicialización
        yield "Iniciando Nexus OS... 🧠\n"
        await asyncio.sleep(0.1)
        
        model = genai.GenerativeModel(settings.GENERATIVE_MODEL)
        kb_memory = get_kb_memory()

        # 1. INVESTIGADOR
        yield "Paso 1: Investigando nodos y patrones... 🕵️\n"
        # Nota: agent_investigator es síncrono, lo ejecutamos directo por ahora.
        investigation = agent_investigator(user_prompt, model, kb_memory)
        
        n_nodes = len(investigation.get("candidate_nodes", []))
        n_cases = len(investigation.get("case_studies", []))
        yield f"   -> Detectados {n_nodes} nodos clave y {n_cases} patrones de referencia.\n"
        await asyncio.sleep(0.1)

        # 2. ARQUITECTO
        yield "Paso 2: Diseñando arquitectura lógica... 🏛️\n"
        logical_plan = agent_architect(investigation, user_prompt, kb_memory, model)
        
        if not logical_plan:
            yield "ERROR: El arquitecto no pudo diseñar el plan. Abortando."
            return

        yield "   -> Plan maestro aprobado.\n"
        await asyncio.sleep(0.1)

        # 3. BUILDER
        yield "Paso 3: Construyendo nodos... 🔨\n"
        nodes, connections = build_nodes_from_plan(logical_plan, kb_memory)
        yield f"   -> Estructura creada con {len(nodes)} nodos.\n"

        # 4. WRITER
        yield "Paso 4: Redactando documentación... 📝\n"
        nodes_with_notes = agent_technical_writer(nodes, user_prompt, model)
        
        # 5. ASSEMBLER
        yield "Paso 5: Ensamblaje final... 📦\n"
        final_json_str = final_assembler(nodes_with_notes, connections, user_prompt)
        
        # Respuesta final para el Frontend (Formato JSON esperado por Streamlit)
        final_output = {
            "workflow_json": json.loads(final_json_str),
            "executive_summary": "Workflow generado por Nexus OS v4.0 (Modular)."
        }
        
        # Enviamos el JSON final en una sola línea para que el frontend lo parsee
        yield json.dumps(final_output)
        
        logger.info("✅ Pipeline completado con éxito.")

    except Exception as e:
        logger.error(f"🔥 Error crítico en pipeline: {e}", exc_info=True)
        yield f"ERROR: Fallo en el sistema: {str(e)}"