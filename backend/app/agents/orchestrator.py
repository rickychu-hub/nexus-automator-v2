import logging
import json
import asyncio
import google.generativeai as genai
from app.core.config import settings

# Importamos los agentes y servicios
from app.services.chroma_service import get_kb_memory
from app.services.database_service import save_workflow_log

# --- IMPORTAMOS EL SERVICIO DE DESPLIEGUE ---
from app.services.n8n_service import n8n_deployer
# ---------------------------------------------------

from app.agents.investigator import agent_investigator
from app.agents.architect import agent_architect
from app.agents.builder import build_nodes_from_plan, final_assembler
from app.agents.writer import agent_technical_writer

logger = logging.getLogger(__name__)

async def stream_generation_pipeline(user_prompt: str):
    """
    Pipeline V5.0 (Headless):
    Investigador -> Arquitecto -> Builder -> Writer -> Assembler -> [Deployer] -> DB
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
        nodes, connections = build_nodes_from_plan(logical_plan, kb_memory, user_prompt, model)
        yield f"   -> Estructura creada con {len(nodes)} nodos (Mock Data Inyectada).\n"

        # 4. WRITER
        yield "Paso 4: Redactando documentación... 📝\n"
        nodes_with_notes, guide_markdown = agent_technical_writer(nodes, user_prompt, model)
        
        # 5. ASSEMBLER
        yield "Paso 5: Ensamblaje final... 📦\n"
        final_json_str = final_assembler(nodes_with_notes, connections, user_prompt)
        
        # Parseamos el JSON para manipularlo
        final_json_obj = json.loads(final_json_str)

        # ==============================================================================
        # PASO 6: INYECCIÓN AUTOMÁTICA (HEADLESS)
        # ==============================================================================
        yield "Paso 6: Inyectando en n8n... 🚀\n"
        
        deployment_data = None
        webhook_msg = ""
        
        try:
            # Llamada al nuevo servicio
            deployment_result = n8n_deployer.deploy_workflow(final_json_obj)
            
            wf_id = deployment_result.get('id')
            webhook_url = deployment_result.get('webhook_url')
            
            # Mensajes de éxito
            msg = f"   -> ✅ Workflow desplegado y activado (ID: {wf_id}).\n"
            if webhook_url:
                msg += f"   -> 🔗 WEBHOOK PÚBLICO: {webhook_url}\n"
                webhook_msg = f"\n\n🔗 **Webhook Activo:** `{webhook_url}`"
            else:
                msg += "   -> ℹ️ No se detectaron Webhooks públicos.\n"
            
            yield msg
            
            # Añadimos los datos del despliegue al JSON final
            final_json_obj['deployment'] = deployment_result
            deployment_data = deployment_result

        except Exception as e:
            # Si falla el despliegue, NO rompemos el pipeline. Solo avisamos.
            logger.error(f"⚠️ Error en despliegue Headless: {e}")
            yield f"   -> ⚠️ Aviso: No se pudo inyectar automáticamente. Entregando JSON manual.\n"
        
        # ==============================================================================
        # PASO 6.5: GENERACIÓN DE TÍTULO INTELIGENTE
        # ==============================================================================
        yield "Generando nombre de sistema Nexus... 🏷️\n"
        smart_title = ""
        
        try:
            # PROMPT AJUSTADO
            naming_prompt = (
                f"Analiza esta solicitud: '{user_prompt}'. "
                f"Tu tarea es generar un Título de Sistema Técnico para este workflow. "
                f"REGLAS OBLIGATORIAS:"
                f"1. Debe empezar SIEMPRE con el prefijo exacto: 'Nexus.OS :: '"
                f"2. Debe usar vocabulario técnico (ej: Ingesta, Despliegue, Sincronización)."
                f"3. Longitud ideal: entre 6 y 12 palabras."
                f"Devuelve SOLO el título final sin comillas."
            )
            
            title_resp = model.generate_content(naming_prompt)
            # Limpieza extra
            smart_title = title_resp.text.strip().replace('"', '').replace("Title:", "")
            logger.info(f"🏷️ Nombre generado: {smart_title}")

        except Exception as e:
            # Fallback por si falla la IA
            smart_title = f"Nexus.OS :: Workflow Automatizado ({user_prompt[:20]}...)"
        
        # [FIX] Sincronización de Nombres
        # Forzamos que el JSON tenga el nombre inteligente para que al importar en n8n salga bien
        if final_json_obj and isinstance(final_json_obj, dict):
            final_json_obj['name'] = smart_title
            # Tambien inyectamos en meta por si acaso
            if 'meta' not in final_json_obj: final_json_obj['meta'] = {}
            final_json_obj['meta']['smart_title'] = smart_title
            logger.info(f"✅ Nombre del JSON actualizado a: {smart_title}")

        # ==============================================================================

        # 7. GUARDAR EN SUPABASE
        yield "Paso 7: Guardando en memoria persistente...\n"
        try:
            # Añadimos metadatos del despliegue al log
            meta_info = "Workflow generado por Nexus"
            if deployment_data:
                meta_info += f" [Auto-Deployed ID: {deployment_data.get('id')}]"

            # Pasamos 'smart_title' a la función
            save_workflow_log(user_prompt, final_json_obj, meta_info, smart_title)
            logger.info("✅ Guardado en base de datos ejecutado.")
        except Exception as e:
            logger.error(f"⚠️ Error al guardar en DB: {e}")
        
        # Respuesta final para el Frontend
        summary_text = "Workflow generado por Nexus OS." + webhook_msg
        
        final_output = {
            "workflow_json": final_json_obj,
            "executive_summary": summary_text,
            "configuration_manual": guide_markdown 
        }
        
        yield json.dumps(final_output)
        
        logger.info("✅ Pipeline completado con éxito.")

    except Exception as e:
        logger.error(f"🔥 Error crítico en pipeline: {e}", exc_info=True)
        yield f"ERROR: Fallo en el sistema: {str(e)}"