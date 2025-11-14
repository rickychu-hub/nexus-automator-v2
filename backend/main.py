# main.py (VERSIÓN FINAL - SOLO CONEXIÓN A CHROMADB)
from fastapi.responses import StreamingResponse
import asyncio # Necesario para los 'yields'
import logging
from fastapi import FastAPI
from pydantic import BaseModel
import json
import os
import re
import time
from datetime import datetime
import google.generativeai as genai
import chromadb
import copy
from dotenv import load_dotenv, find_dotenv
from chromadb.utils import embedding_functions # Asegurar que esté importado
import asyncio



# --- CONFIGURACIÓN ---
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
API_KEY = os.getenv("GOOGLE_API_KEY")
print(f"🔍 API_KEY leída (primeros 5 chars): {API_KEY[:5] if API_KEY else 'NO ENCONTRADA'}")

EMBEDDING_MODEL = 'models/embedding-001'
GENERATIVE_MODEL = 'models/gemini-2.5-flash' # Mantenemos 'pro' por estabilidad
ENRICHED_KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base_final_CURATED.json")
CHROMA_DB_PATH = os.getenv("CHROMA_PERSIST_PATH", "/data/chroma_db_v2")

ENCYCLOPEDIA_COLLECTION = 'n8n_nodes_final_v5'
EXPERIENCE_COLLECTION = 'n8n_workflow_cases_v1'

# --- Inicialización Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuración de Google Generative AI ---
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        logger.info("Google Generative AI configurado.")
    except Exception as e:
        logger.error(f"Error configurando Google Generative AI: {e}", exc_info=True)
else:
    logger.error("¡¡¡ERROR CRÍTICO!!! GOOGLE_API_KEY no encontrada.")

# --- Función de Embeddings (Solo definición, no se usa en main ahora) ---
# Necesitamos la clase aquí si la usamos en get_collection, aunque no genere embeddings al inicio
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        super().__init__()
    # La implementación __call__ no es estrictamente necesaria aquí si solo conectamos
    # Pero la dejamos por si Chroma la necesita internamente al obtener la colección
    def __call__(self, texts):
        # Implementación mínima o la completa si Chroma la requiere
        logger.debug(f"GeminiEmbeddingFunction.__call__ invocada para {len(texts)} textos (solo conexión)")
        # Devolver vectores nulos podría ser suficiente si solo es para obtener la colección
        # o llamar a la implementación real si Chroma falla sin ella.
        # Por seguridad, mantenemos la llamada real pero con logging.
        embeddings = []
        batch_size = 100
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            # ... (Implementación completa de __call__ que tenías) ...
            current_batch_num = i // batch_size + 1
            batch_texts = texts[i:i + batch_size]
            processed_batch = []
            # ... (truncado) ...
            try:
                # ... (llamada a genai.embed_content) ...
                result = genai.embed_content(model=EMBEDDING_MODEL, content=processed_batch, task_type="retrieval_document")
                embeddings.extend(result['embedding'])
            except Exception as e:
                logger.error(f"Error embeddings en __call__ (conexión): {e}")
                embeddings.extend([[0.0] * 768] * len(batch_texts))
            time.sleep(0.5) # Pausa más corta
        return embeddings


# --- Variables Globales para ChromaDB ---
chroma_client = None
kb_collection = None
exp_collection = None




# --- CARGA DE BASE DE CONOCIMIENTO EN MEMORIA (Rápida) ---
knowledge_base_global = {}
try:
    if os.path.exists(ENRICHED_KB_PATH):
        with open(ENRICHED_KB_PATH, 'r', encoding='utf-8') as f:
            kb_original = json.load(f)

            # --- ¡AQUÍ ESTÁ EL ARREGLO V2.5! ---
            # Inyectamos el 'type' (con mayúsculas/minúsculas correctas)
            # DENTRO del template, ANTES de hacer la clave minúscula.
            knowledge_base_global = {}
            for k, v in kb_original.items():
                v['type'] = k  # Inyecta el 'type' (ej: "n8n-nodes-base.googleSheets")
                knowledge_base_global[k.lower()] = v # La clave sigue siendo minúscula

            logger.info(f"Base de conocimiento en memoria cargada ({len(knowledge_base_global)} nodos) e inyectada con 'type'.")
# ... (etc)
    else:
         logger.error(f"{ENRICHED_KB_PATH} no encontrado para carga en memoria.")
except Exception as e:
    logger.error(f"Error cargando KB en memoria: {e}", exc_info=True)


# --- AGENTES (Definiciones completas) ---

# AGENTE ENTREVISTADOR (Simplificado Anti-Bucle)
# -----------------------------------------------------------------
# REEMPLAZA TU FUNCIÓN 'agent_interviewer' ANTIGUA POR ESTA COMPLETA
# -----------------------------------------------------------------
def agent_interviewer(original_prompt, questions, answers, model):
    logger.info("Iniciando Agente Entrevistador (V2.3 con consolidación segura)...")
    conversation_history = ""
    has_history = False
    if questions and answers:
        history_parts = []
        for q, a in zip(questions, answers):
            if a and a.strip():
                history_parts.append(f"- Pregunta anterior: {q}\n- Respuesta del usuario: {a}")
        if history_parts:
            conversation_history = "\n".join(history_parts)
            has_history = True

    # --- Rama 1: Consolidar historial (¡AQUÍ ESTÁ EL ARREGLO V2.3!) ---
    if has_history:
        logger.info("Detectado historial de respuestas. Forzando consolidación.")
        # NUEVO PROMPT: Pide SOLO el texto del briefing, no el JSON.
        prompt = (f"Actúas como un consultor de automatización. El usuario ha respondido a tus preguntas previas. Tu ÚNICA tarea ahora es consolidar la petición original y TODAS las respuestas del historial en un briefing técnico final y detallado para un ingeniero.\n\n"
                  f"**Petición Original:**\n\"{original_prompt}\"\n\n"
                  f"**Historial de la Conversación:**\n{conversation_history}\n\n"
                  f"**Instrucción Clave:** Genera el briefing MÁS COMPLETO POSIBLE con la información disponible. NO HAGAS MÁS PREGUNTAS.\n\n"
                  f"**Formato de Salida Obligatorio:**\n"
                  f"Responde ÚNICAMENTE con el texto del briefing consolidado. NO incluyas '```json' ni nada más.")
        
        try:
            if not isinstance(model, genai.GenerativeModel):
                 logger.error("El objeto 'model' no es una instancia válida de GenerativeModel.")
                 return {"status": "clarified", "briefing": original_prompt + "\n" + conversation_history} # Fallback

            response = model.generate_content(prompt)
            briefing_text = response.text.strip().replace('```', '') # Obtenemos el texto plano y quitamos ``` por si acaso

            # ¡Construimos el JSON nosotros mismos! 100% seguro.
            result_json = {
                "status": "clarified",
                "briefing": briefing_text
            }
            logger.info(f"Agente Entrevistador consolidó el briefing.")
            return result_json # Devolvemos el JSON construido en Python

        except Exception as e:
            logger.error(f"Error en Agente Entrevistador (consolidación): {e}", exc_info=True)
            briefing = original_prompt + "\n" + conversation_history
            return {"status": "clarified", "briefing": briefing} # Fallback de seguridad

    # --- Rama 2: Primera ronda (80/20) (Esta parte se queda igual) ---
    else:
        logger.info("Primera ronda de la entrevista. Aplicando estrategia 80/20.")
        prompt = (f"Actúas como un consultor de automatización eficiente (Regla 80/20). Analiza la siguiente petición INICIAL. Tu objetivo es obtener solo los 3 puntos clave: Trigger, Aplicaciones Principales, Lógica Central. NO pidas detalles finos (nombres de archivo/hoja, columnas, etc.).\n\n"
                  f"**Petición Original:**\n\"{original_prompt}\"\n\n"
                  f"**Tu Proceso:**\n1. ¿La petición inicial cubre claramente los 3 puntos clave?\n2. Si SÍ, genera directamente el briefing.\n3. Si NO, haz 1 o 2 preguntas MÁXIMO para obtener SOLO la información esencial faltante.\n\n"
                  f"**Formato de Salida Obligatorio (SOLO JSON):**\n\n* **Si necesitas MÁS INFORMACIÓN:**\n```json\n{{\n    \"status\": \"needs_more_info\",\n    \"questions\": [\"Tu pregunta específica aquí...\"]\n}}\n```\n\n* **Si la información es SUFICIENTE:**\n```json\n{{\n    \"status\": \"clarified\",\n    \"briefing\": \"Aquí va el resumen técnico inicial.\"\n}}\n```")
        
        try:
            if not isinstance(model, genai.GenerativeModel):
                 logger.error("El objeto 'model' no es una instancia válida de GenerativeModel.")
                 return {"status": "clarified", "briefing": original_prompt} # Fallback
            response = model.generate_content(prompt)
            text_response = response.text.strip()
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_response, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group(1))
                logger.info(f"Agente Entrevistador devolvió JSON: {result_json}")
                return result_json
            else:
                logger.warning("El Agente Entrevistador no devolvió un JSON válido. Forzando clarificación.")
                return {"status": "clarified", "briefing": original_prompt}
        except Exception as e:
            logger.error(f"Error en Agente Entrevistador (primera ronda): {e}", exc_info=True)
            return {"status": "clarified", "briefing": original_prompt}


# AGENTE INVESTIGADOR (Usa globales)
def agent_investigator(user_request, model, knowledge_base): # <-- CAMBIOS AQUÍ
    logger.info("Iniciando Agente Investigador (Estratégico V2.2)...")
    global kb_collection, exp_collection # Usará las globales de ChromaDB

    if not kb_collection or not exp_collection:
         logger.error("Colecciones ChromaDB globales no están disponibles.")
         return {"candidate_nodes": [], "case_studies": []}

    # --- PASO 1: El Investigador "Piensa" ---
    # Le pedimos a la IA que valide la petición contra nuestra Enciclopedia
    all_node_names = list(knowledge_base.keys()) # Lista de los 794 nodos

    prompt = (
        f"Actúas como un Analista de n8n experto. Tu ÚNICA tarea es leer un briefing y compararlo con una lista de todos los nodos disponibles para encontrar los `nodeId`s EXACTOS necesarios.\n\n"
        f"**Briefing del Usuario:**\n\"{user_request}\"\n\n"
        f"**Lista de Nodos Disponibles (Enciclopedia Completa):**\n{json.dumps(all_node_names, indent=2)}\n\n"
        f"**Tu Proceso Mental:**\n"
        f"1. ¿Qué nodos pide el briefing? (Ej. 'Tally', 'Google Sheets', 'IF').\n"
        f"2. Para cada nodo, ¿existe un `nodeId` EXACTO en la Enciclopedia? (Ej. 'n8n-nodes-base.googleSheets', 'n8n-nodes-base.if').\n"
        f"3. Si una app (como 'Tally') NO existe en la lista, ¿cuál es el nodo GENÉRICO correcto para esa función? (Ej. 'Tally' es un formulario, el nodo genérico es 'n8n-nodes-base.webhook').\n"
        f"4. Devuelve una lista JSON SOLO con los `nodeId`s validados y correctos.\n\n"
        f"**Formato de Salida Obligatorio (SOLO JSON):**\n"
        f"```json\n"
        f"{{\n"
        f'  "required_nodes": ["nodeId-real-1", "nodeId-real-2", "nodeId-generico-si-es-necesario"]\n'
        f"}}\n"
        f"```"
    )

    validated_node_ids = []
    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_response, re.DOTALL)
        if json_match:
            result_json = json.loads(json_match.group(1))
            validated_node_ids = result_json.get("required_nodes", [])
            logger.info(f"Investigador (Fase Pensar) validó estos nodos: {validated_node_ids}")
        else:
            logger.warning("Investigador (Fase Pensar) no devolvió JSON. Se usará la búsqueda semántica simple.")
            # Fallback a la V1 (búsqueda semántica) si el "pensar" falla
            return agent_investigator_v1_fallback(user_request) # Necesitaremos crear esta función de fallback

    except Exception as e:
        logger.error(f"Error en Fase Pensar del Investigador: {e}", exc_info=True)
        return agent_investigator_v1_fallback(user_request) # Fallback

    if not validated_node_ids:
         logger.warning("La Fase Pensar no devolvió nodos. Usando fallback.")
         return agent_investigator_v1_fallback(user_request) # Fallback

    # --- PASO 2: El Investigador "Busca Experiencia" (con Foco) ---
    # Ahora busca en la Experiencia (25k docs) patrones sobre esos nodos validados

    # Creamos un texto de búsqueda enfocado
    query_text_v2 = f"Workflow que usa: {', '.join(validated_node_ids)}. Petición original: {user_request}"

    try:
        query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=query_text_v2, task_type="retrieval_query")['embedding']
    except Exception as e:
        logger.error(f"Error generando embedding V2 para consulta: {e}", exc_info=True)
        return {"candidate_nodes": validated_node_ids, "case_studies": []} # Devolver al menos los nodos

    case_studies = []
    try:
        experience_results = exp_collection.query(query_embeddings=[query_embedding], n_results=5, include=['documents'])
        if experience_results and experience_results.get('documents') and experience_results['documents'][0]:
            case_studies = experience_results['documents'][0]
            logger.info(f"Investigador (Fase Buscar) encontró {len(case_studies)} casos V2.")
        else: 
            logger.warning("Investigador (Fase Buscar) no encontró documentos V2 válidos.")
    except Exception as e: 
        logger.error(f"Error búsqueda Experiencia V2: {e}", exc_info=True)

    # Entregamos la lista VALIDADA y los casos relevantes
    return {"candidate_nodes": validated_node_ids, "case_studies": case_studies}


# ----------------------------------------------------
def agent_investigator_v1_fallback(user_request):
    logger.info("Iniciando Agente Investigador (FALLBACK V2.1)...") 
    global kb_collection, exp_collection # <-- Declara que usará las globales

    if not kb_collection or not exp_collection:
         logger.error("Colecciones ChromaDB globales no están disponibles.")
         return {"candidate_nodes": [], "case_studies": []}

    try:
        query_embedding = genai.embed_content(model=EMBEDDING_MODEL, content=user_request, task_type="retrieval_query")['embedding']
    except Exception as e:
        logger.error(f"Error generando embedding para consulta: {e}", exc_info=True)
        return {"candidate_nodes": [], "case_studies": []}

    candidate_node_ids, case_studies = [], []
    try:
        # Esta es la lógica V2.1 que busca 'metadatas'
        encyclopedia_results = kb_collection.query(query_embeddings=[query_embedding], n_results=30, include=['metadatas'])
        if encyclopedia_results and encyclopedia_results.get('metadatas') and encyclopedia_results['metadatas'][0]:
             candidate_node_ids = [meta.get('node_id', meta.get('id', '')) for meta in encyclopedia_results['metadatas'][0] if meta]
             candidate_node_ids = [nid for nid in candidate_node_ids if nid] # Limpiar nulos
             logger.info(f"Enciclopedia (Fallback): {len(candidate_node_ids)} candidatos.")
        else: 
            logger.warning("Investigador (Fallback) no encontró 'metadatas' válidas.")
            candidate_node_ids = [] # Asegurar lista vacía
    except Exception as e: 
        logger.error(f"Error búsqueda Enciclopedia (Fallback): {e}", exc_info=True)
        candidate_node_ids = [] # Asegurar lista vacía

    try:
        experience_results = exp_collection.query(query_embeddings=[query_embedding], n_results=5, include=['documents'])
        if experience_results and experience_results.get('documents') and experience_results['documents'][0]:
            case_studies = experience_results['documents'][0]
            logger.info(f"Experiencia (Fallback): {len(case_studies)} casos.")
        else: logger.warning("Sin documentos válidos en Experiencia (Fallback).")
    except Exception as e: logger.error(f"Error búsqueda Experiencia (Fallback): {e}", exc_info=True)

    return {"candidate_nodes": candidate_node_ids, "case_studies": case_studies}

# AGENTE ARQUITECTO
def agent_architect(investigation_results, user_request, knowledge_base, model):
    logger.info("Iniciando Agente Arquitecto...")
    candidate_node_ids = investigation_results.get("candidate_nodes", [])
    case_studies = investigation_results.get("case_studies", [])
    if not candidate_node_ids:
         logger.warning("Agente Arquitecto: lista vacía de nodos candidatos.")

    candidate_details = []
    for nid in candidate_node_ids:
         node_info = knowledge_base.get(nid.lower()) # KB ya está en minúsculas
         if node_info:
             candidate_details.append({
                 "nodeId_EXACTO_A_USAR": nid, # Usar ID original para el plan
                 "descripcion": node_info.get('description', 'Sin descripción.')
             })
         else:
              logger.warning(f"Nodo '{nid}' no encontrado en KB en memoria.")

    # --- INICIO DE CAMBIOS ---
    # 1. El prompt se actualiza para pedir el plan Y las pruebas
    prompt = (
        f"Actúas como Arquitecto n8n de élite. Tu trabajo es diseñar un plan lógico Y crear pruebas de validación.\n\n"
        f"**Petición:** \"{user_request}\"\n"
        f"**Casos:**\n```json\n{json.dumps(case_studies, indent=2, ensure_ascii=False)}\n```\n"
        f"**Nodos Disponibles:**\n"
        f"```json\n{json.dumps(candidate_details, indent=2, ensure_ascii=False)}\n```\n"
        
        f"--- TAREA 1: El Plan Lógico ---\n"
        f"Diseña el plan de nodos. PARA CADA NODO, debes incluir:\n"
        f"1. `nodeId`: El ID exacto del nodo (ej: 'n8n-nodes-base.zohoCrm').\n"
        f"2. `unique_id`: Un nombre clave único para este paso (ej: 'buscar_lead', 'crear_contacto').\n"
        f"3. `purpose`: Una descripción clara de la intención (ej: 'Buscar un Lead existente', 'Crear un nuevo Contacto').\n\n"
        
        f"--- TAREA 2: Las Pruebas de Validación ---\n"
        f"Crea pruebas para los pasos CRÍTICOS (especialmente nodos como Zoho, Salesforce, etc.).\n"
        f"1. `unique_id_to_test`: El `unique_id` del nodo que quieres probar.\n"
        f"2. `parameter_to_check`: La ruta del parámetro que debe ser correcto (ej: 'resource', 'operation').\n"
        f"3. `expected_value`: El valor que DEBE tener (ej: 'contact', 'lead', 'create').\n\n"
        
        f"--- Formato de Salida Obligatorio (SOLO JSON Objeto) ---\n"
        f"```json\n"
        f"{{\n"
        f'  "logical_plan": [\n'
        f'    {{"nodeId": "n8n-nodes-base.zohoCrm", "unique_id": "buscar_lead", "purpose": "Buscar Lead existente."}},\n'
        f'    {{"nodeId": "n8n-nodes-base.if", "unique_id": "si_no_hay_lead", "purpose": "Si no se encuentra Lead..."}},\n'
        f'    {{"nodeId": "n8n-nodes-base.zohoCrm", "unique_id": "crear_contacto", "purpose": "Crear nuevo Contacto."}}\n'
        f'  ],\n'
        f'  "validation_tests": [\n'
        f'    {{"unique_id_to_test": "buscar_lead", "parameter_to_check": "resource", "expected_value": "lead"}},\n'
        f'    {{"unique_id_to_test": "crear_contacto", "parameter_to_check": "resource", "expected_value": "contact"}},\n'
        f'    {{"unique_id_to_test": "crear_contacto", "parameter_to_check": "operation", "expected_value": "create"}}\n'
        f'  ]\n'
        f'}}\n'
        f"```"
    )
    
    try:
        if not isinstance(model, genai.GenerativeModel): raise TypeError("Modelo IA no válido")
        response = model.generate_content(prompt)
        
        # 2. El 're.search' ahora SOLO busca un objeto {}. Se elimina el fallback a [].
        json_str_match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', response.text, re.DOTALL)
        
        if json_str_match:
            # 3. La lógica de parseo y retorno se actualiza para manejar el nuevo objeto
            architect_output = json.loads(json_str_match.group(1))
            logger.info(f"Plan de arquitecto generado:\n{json.dumps(architect_output, indent=2)}")
            
            # Validamos que el output tenga las claves que esperamos
            if not isinstance(architect_output, dict) or "logical_plan" not in architect_output or "validation_tests" not in architect_output:
                 logger.error(f"El arquitecto no devolvió la estructura 'logical_plan' y 'validation_tests'.")
                 return None
            
            return architect_output # Devolvemos el objeto completo
        else:
            logger.error(f"Arquitecto no devolvió JSON válido. Respuesta: {response.text}")
            return None
    except Exception as e:
        logger.error(f"Error en Agente Arquitecto: {e}", exc_info=True)
        return None
    # --- FIN DE CAMBIOS ---
# AGENTE REDACTOR TÉCNICO
def agent_technical_writer(nodes_to_document, user_request, model):
    # ... (Código completo de agent_technical_writer que ya tenías)
    logger.info("Iniciando Agente Redactor Técnico (Manual de Vuelo)...")
    if not nodes_to_document:
        logger.warning("Lista de nodos para documentar está vacía.")
        return []
    workflow_summary = [f"- Paso {i+1}: {node.get('name')} ({node.get('type')})" for i, node in enumerate(nodes_to_document)]
    workflow_plan_str = "\n".join(workflow_summary)

    for node in nodes_to_document:
        node_type = node.get('type', '')
        node_name = node.get('name', 'NodoDesconocido')
        if 'Trigger' in node_type:
             node['instructions'] = ("**Propósito:** Iniciar el flujo automáticamente.\n"
                                     "**Tareas autocompletadas:** Nodo configurado.\n"
                                     "**Tareas pendientes:** Verifica credenciales y evento.\n"
                                     "**Consejo:** Prueba en modo manual.")
             logger.info(f"Nota estándar para Trigger: {node_name}")
             continue

        prompt = (f"Eres Asistente n8n. Redacta nota práctica para nodo '{node_name}' ({node_type}).\n"
                  f"**Petición:** {user_request}\n"
                  f"**Plan:**\n{workflow_plan_str}\n"
                  f"**Config IA (LOS PARÁMETROS QUE HE RELLENADO):**\n{json.dumps(node.get('parameters', {}), indent=2)}\n\n"
                  f"--- FORMATO OBLIGATORIO (texto plano) ---\n"
                  f"**Propósito:** [Objetivo en 1 frase]\n"
                  f"**Tareas autocompletadas:** [Qué configuró IA (ej. expresiones, lógica)]\n"
                  f"**Tareas pendientes para ti:** [¡¡IMPORTANTE!! Analiza la 'Config IA' de arriba. Si ves CUALQUIER valor como 'YOUR_..._HERE' o similar, enumera explícitamente CADA campo que el usuario debe rellenar manualmente (ej. 'Rellena el spreadsheetId', 'Configura las credenciales'). Si no hay nada pendiente, escribe 'Ninguna.'.]\n"
                  f"**Consejo del Co-Piloto:** [Tip corto]\n")
        try:
            if not isinstance(model, genai.GenerativeModel): raise TypeError("Modelo IA no válido")
            response = model.generate_content(prompt)
            node['instructions'] = response.text.strip()
            logger.info(f"Nota generada para '{node_name}'")
        except Exception as e:
            node['instructions'] = f"Error generando nota: {e}"
            logger.error(f"Error generando nota para '{node_name}': {e}", exc_info=True)
        time.sleep(1) # Pausa
    return nodes_to_document


# --- ¡NUEVO AGENTE DE VALIDACIÓN! ---
def agent_validator(nodes_with_params, validation_tests):
    """
    Comprueba el trabajo del Configurador contra las pruebas del Arquitecto.
    Esto es el Bucle de Validación Interna, implementado como Python puro.
    """
    logger.info("Iniciando Agente Validador (Control de Calidad)...")
    errors = []
    
    if not validation_tests:
        logger.warning("Validador no recibió pruebas del Arquitecto. Omitiendo.")
        return {"status": "passed"} # No hay pruebas, así que pasa

    # Mapear nodos por su ID único para fácil acceso
    nodes_map = {
        node.get('unique_id_from_architect'): node 
        for node in nodes_with_params 
        if 'unique_id_from_architect' in node
    }

    for test in validation_tests:
        test_id = test.get('unique_id_to_test')
        param_path = test.get('parameter_to_check')
        expected_value = test.get('expected_value')

        if not test_id or not param_path or expected_value is None:
            logger.warning(f"Prueba de validación mal formada: {test}")
            continue

        node_to_check = nodes_map.get(test_id)

        if not node_to_check:
            errors.append(f"Prueba fallida: El plan requería un paso '{test_id}', pero no se encontró.")
            continue
        
        # Navegar por el JSON de parámetros (ej: 'resource' o 'options.operation')
        try:
            actual_value = node_to_check['parameters']
            for key in param_path.split('.'):
                actual_value = actual_value[key]
        except (KeyError, TypeError):
            actual_value = None # No se encontró el parámetro

        if actual_value != expected_value:
            errors.append(
                f"Prueba Lógica fallida en nodo '{node_to_check.get('name', test_id)}':\n"
                f"  > El Arquitecto esperaba que '{param_path}' fuera '{expected_value}'.\n"
                f"  > Pero el Configurador lo estableció como '{actual_value}'.\n"
            )

    if errors:
        logger.error(f"Validación interna FALLIDA: {len(errors)} errores encontrados.")
        return {"status": "failed", "errors": errors}
    
    logger.info("✅ Validación interna superada. El flujo es lógicamente correcto.")
    return {"status": "passed"}
# --- FIN DEL NUEVO AGENTE ---
# AGENTE CONFIGURADOR
def agent_parameter_configurator(nodes, user_request, investigation_results, model, knowledge_base):
    logger.info("Iniciando Agente Configurador...")
    case_studies = investigation_results.get("case_studies", [])
    if not nodes:
         logger.warning("Lista de nodos para configurar está vacía.")
         return []
    
    for i, node in enumerate(nodes):
        node_type = node.get('type', '')
        node_name = node.get('name', f"nodo_{i+1}")
        
        if 'Trigger' in node_type:
            logger.info(f"Saltando Trigger: {node_name}")
            continue
            
        node_id_key = node_type.lower()
        node_manual_data = knowledge_base.get(node_id_key, {})
        configurable_params = node_manual_data.get('properties')
        
        if not configurable_params:
            logger.warning(f"Sin propiedades en KB para '{node_name}' ({node_id_key}).")
            node['parameters'] = {}
            continue
            
        context_summary = []
        prev_node_name = None
        if i > 0 and nodes[i-1]:
             prev_node = nodes[i-1]
             prev_node_name = prev_node.get('name')
             context_summary.append(f"Paso {i}: {prev_node_name} ({prev_node.get('type')})")
        else: context_summary.append("Primer nodo post-Trigger.")
        workflow_context = "\n".join(context_summary)
        
        # --- INICIO DE CAMBIOS ---
        # 1. Obtenemos el 'purpose' inyectado por el Builder
        node_purpose = node.get('purpose', 'Sin propósito claro.')
        # --- FIN DE CAMBIOS ---

        last_error = ""
        for attempt in range(2):
            expression_example = ""
            if prev_node_name: expression_example = f" (ej: `{{{{$node[\"{prev_node_name}\"].json.id}}}}`)"
            
            # --- INICIO DE CAMBIOS ---
            # 2. El prompt se actualiza para incluir el 'purpose' y la 'INSTRUCCIÓN CRÍTICA'
            prompt = (
                f"Configura parámetros JSON para nodo '{node_name}' ({node_type}).\n"
                f"**Petición General:** {user_request}\n"
                f"**Nodo Anterior:** {workflow_context}\n"
                f"**Propósito Específico de este Nodo:** \"{node_purpose}\"\n\n"
                
                f"**¡¡INSTRUCCIÓN CRÍTICA!!** Usa el `Propósito Específico` para decidir los parámetros.\n"
                f"* Si el propósito es 'Crear nuevo Contacto', DEBES usar `\"resource\": \"contact\"` y `\"operation\": \"create\"`.\n"
                f"* Si el propósito es 'Buscar Lead existente', DEBES usar `\"resource\": \"lead\"` y `\"operation\": \"get\"`.\n"
                f"* Si el propósito es 'Crear nuevo Lead', DEBES usar `\"resource\": \"lead\"` y `\"operation\": \"create\"`.\n\n"
                
                f"**Guía KB (Propiedades del Nodo):**\n{json.dumps(configurable_params, indent=2)}\n"
                f"**Ejemplos (Contexto de otros flujos):**\n{json.dumps(case_studies, indent=2)}\n"
                f"{last_error}\n\n"
                f"Devuelve SOLO JSON con parámetros probables. Usa expresiones `{{{{...}}}}`{expression_example} para datos previos. NO incluyas credenciales/IDs."
            )
            # --- FIN DE CAMBIOS ---
            
            try:
                if not isinstance(model, genai.GenerativeModel): raise TypeError("Modelo IA no válido")
                response = model.generate_content(prompt)
                json_str_match = re.search(r'```json\s*(\{.*?\})\s*```', response.text, re.DOTALL)
                
                if not json_str_match: 
                    params_json = json.loads(response.text.strip())
                else: 
                    params_json = json.loads(json_str_match.group(1))
                    
                if not isinstance(params_json, dict): raise ValueError("Respuesta no es diccionario JSON.")
                
                node['parameters'] = params_json
                logger.info(f"Parámetros configurados '{node_name}' (Intento {attempt+1})")
                break
                
            except Exception as e:
                logger.warning(f"Error config '{node_name}' (Intento {attempt+1}): {e}")
                last_error = f"Intento anterior fallido ({e}). Revisa JSON/expresiones."
                if attempt == 1: node['parameters'] = {"error_config_auto": f"{e}"}
                
        time.sleep(1) # Mantenemos la pausa
        
    return nodes


# BUILDER (V5.0 - Arreglo de "Nodo Huérfano")
def build_nodes_from_plan(logical_plan, knowledge_base):
    """
    Construye la estructura de nodos y conexiones a partir del plan lógico del Arquitecto.
    V5.0: Arregla el bug de encadenamiento de nodos en ramas (ej. IF -> SendGrid -> Slack).
    """
    if not isinstance(logical_plan, list):
        logger.error("build_nodes_from_plan recibió un plan inválido.")
        return [], {}

    logger.info("Construyendo estructura desde plan...")
    nodes, connections, node_counts = [], {}, {}

    def process_plan_recursive(plan, parent_node_name=None, branch_type=None):
        """
        Función auxiliar recursiva para procesar el plan anidado.
        """
        nonlocal node_counts

        # last_node_in_chain rastrea el nodo anterior DENTRO de este bucle
        # Inicia siendo el nodo padre (ej. el 'IF' o el 'Webhook')
        last_node_in_chain = parent_node_name 

        for i, step in enumerate(plan):
            node_id = step.get('nodeId')
            
            # --- INICIO DE CAMBIOS ---
            # 1. Obtenemos el 'purpose' y 'unique_id' del plan del Arquitecto
            node_purpose = step.get('purpose', 'Sin propósito definido')
            node_unique_id = step.get('unique_id', f'step_{len(nodes)}')
            # --- FIN DE CAMBIOS ---
            
            if not node_id: continue

            node_id_lower = node_id.lower().strip()
            node_template = copy.deepcopy(knowledge_base.get(node_id_lower))

            if not node_template:
                logger.warning(f"Nodo '{node_id}' no encontrado en KB. Omitiendo.")
                continue

            # Crear nombre único (ej. googlesheets_1, googlesheets_2)
            base_name = node_template.get('name', node_id.split('.')[-1]) # Usa el 'name' de KB
            count = node_counts.get(base_name, 0) + 1
            node_counts[base_name] = count
            current_node_name = f"{base_name}_{count}"

            # Actualizar el template del nodo
            node_template['id'] = f"node_{len(nodes)}" # ID único temporal de Streamlit
            node_template['name'] = current_node_name
            
            # --- INICIO DE CAMBIOS ---
            # 2. Inyectamos los nuevos datos en el objeto del nodo
            node_template['purpose'] = node_purpose
            node_template['unique_id_from_architect'] = node_unique_id
            # --- FIN DE CAMBIOS ---
            
            nodes.append(node_template)

            # --- ¡LÓGICA DE CONEXIÓN V5.0! ---
            if last_node_in_chain:

                # Decidimos CÓMO conectar:
                # Es una "conexión de rama" (ej. IF -> Nodo) si:
                # 1. Es el primer nodo del bucle (i == 0)
                # 2. Y se nos pasó un 'branch_type' (es decir, venimos de un IF)
                is_first_node_in_branch = (i == 0) and branch_type

                if is_first_node_in_branch:
                    # Conexión especial (Rama True/False)
                    branch_index = 0 if branch_type == 'true' else 1
                    connections.setdefault(last_node_in_chain, {"main": [[], []]})

                    # Asegurarse de que ambas ramas existan
                    while len(connections[last_node_in_chain]["main"]) <= branch_index:
                        connections[last_node_in_chain]["main"].append([])

                    connections[last_node_in_chain]["main"][branch_index].append({"node": current_node_name, "type": "main"})
                    logger.debug(f"Conectando rama '{branch_type}': '{last_node_in_chain}' -> '{current_node_name}'")

                else:
                    # Conexión estándar (Lineal o en Cadena)
                    # (ej. Webhook -> Sheets, o SendGrid -> Slack)
                    connections.setdefault(last_node_in_chain, {"main": [[]]})

                    if not connections[last_node_in_chain]["main"]:
                        connections[last_node_in_chain]["main"].append([]) # Asegurar que main[0] existe

                    connections[last_node_in_chain]["main"][0].append({"node": current_node_name, "type": "main"})
                    logger.debug(f"Conectando lineal (en cadena): '{last_node_in_chain}' -> '{current_node_name}'")

            # ACTUALIZAMOS el "último nodo" para la SIGUIENTE iteración del bucle
            last_node_in_chain = current_node_name

            # --- Recursión para ramas anidadas ---
            if 'branches' in step and isinstance(step['branches'], dict):
                for branch, sub_plan in step['branches'].items():
                    if isinstance(sub_plan, list):
                        # El 'parent_node_name' para la recursión es este nodo actual
                        process_plan_recursive(sub_plan, parent_node_name=current_node_name, branch_type=branch)
                    else: 
                        logger.warning(f"Rama '{branch}' nodo '{current_node_name}' inválida.")

    # Iniciar el proceso recursivo con el plan de nivel superior
    process_plan_recursive(logical_plan)

    logger.info("✅ Estructura pre-construida.")
    return nodes, connections

# ASSEMBLER
def final_assembler(nodes_with_params, connections, user_request):
    # ... (Código completo de final_assembler que ya tenías)
    if not isinstance(nodes_with_params, list):
        logger.error("final_assembler recibió 'nodes_with_params' inválido.")
        nodes_with_params = []
    if not isinstance(connections, dict):
         logger.error("final_assembler recibió 'connections' inválido.")
         connections = {}
    logger.info("Ensamblando workflow final...")
    new_notes, max_note_height = [], 0
    current_note_x, NOTE_Y_START, NOTE_X_SPACING, FIXED_NOTE_WIDTH = 250, 20, 20, 300
    COLOR_PALETTE = ["#A5D6A7", "#FFCC80", "#90CAF9", "#B39DDB", "#F48FB1", "#80CBC4"]
    for i, node in enumerate(nodes_with_params):
        node_id = node.get('id', f'temp_{i}')
        content = f"**NODO: {node.get('name')}**\n\n{node.get('instructions', 'Sin instrucciones.')}"
        dynamic_height = min(400, len(content.split('\n')) * 18 + 50)
        max_note_height = max(max_note_height, dynamic_height)
        new_note = {"id": f"note_for_{node_id}", "type": "n8n-nodes-base.stickyNote", "typeVersion": 1, "name": f"Info {node.get('name')}", "parameters": {"content": content, "color": COLOR_PALETTE[i % len(COLOR_PALETTE)], "width": FIXED_NOTE_WIDTH, "height": dynamic_height}, "position": [current_note_x, NOTE_Y_START]}
        new_notes.append(new_note)
        current_note_x += FIXED_NOTE_WIDTH + NOTE_X_SPACING
    node_positions = {}
    X_START, Y_START_NODES, X_SPACING, Y_SPACING = 250, NOTE_Y_START + max_note_height + 100, 350, 150
    all_node_names_in_plan = {n['name'] for n in nodes_with_params}
    nodes_with_inputs = set()
    for _source_node, conn_data in connections.items():
         if isinstance(conn_data, dict) and "main" in conn_data:
              for branch in conn_data["main"]:
                   if isinstance(branch, list):
                       for target in branch:
                            if isinstance(target, dict) and "node" in target:
                                nodes_with_inputs.add(target["node"])
    start_nodes = [name for name in all_node_names_in_plan if name not in nodes_with_inputs]
    processed_positions = set()
    current_y_offsets = {}
    def position_nodes_recursive(node_name, x, y_level):
        if node_name in processed_positions: return
        y = Y_START_NODES + y_level * Y_SPACING + current_y_offsets.get(y_level, 0)
        node_positions[node_name] = [x, y]
        processed_positions.add(node_name)
        current_y_offsets[y_level] = current_y_offsets.get(y_level, 0) + Y_SPACING / 2
        if node_name in connections:
             all_branches = connections[node_name].get("main", [])
             if len(all_branches) > 0 and isinstance(all_branches[0], list) and all_branches[0]:
                  next_node_name = all_branches[0][0].get('node')
                  if next_node_name: position_nodes_recursive(next_node_name, x + X_SPACING, y_level)
             if len(all_branches) > 1 and isinstance(all_branches[1], list) and all_branches[1]:
                  next_node_name = all_branches[1][0].get('node')
                  if next_node_name: position_nodes_recursive(next_node_name, x + X_SPACING, y_level + 1)
    if start_nodes: position_nodes_recursive(start_nodes[0], X_START, 0)
    else: logger.warning("Sin nodos iniciales para posicionamiento.")
    for node in nodes_with_params:
         if node['name'] in node_positions: node['position'] = node_positions[node['name']]
         elif 'position' not in node: node['position'] = [X_START - 200, Y_START_NODES]
    final_nodes_cleaned = []
    required_keys = ["parameters", "name", "type", "typeVersion", "position", "id", "credentials"]
    all_nodes_final = nodes_with_params + new_notes
    for node in all_nodes_final:
        node['id'] = str(node.get('id', f"missing_id_{time.time()}"))
        clean_node = {key: node[key] for key in required_keys if key in node}
        if 'parameters' not in clean_node: clean_node['parameters'] = {}
        final_nodes_cleaned.append(clean_node)
    final_workflow = {"name": user_request[:60].replace('\n',' '), "nodes": final_nodes_cleaned, "connections": connections, "active": False, "settings": {}, "staticData": None}
    logger.info("✅ Workflow final ensamblado.")
    return json.dumps(final_workflow, indent=2, ensure_ascii=False)


# --- ORQUESTADOR PRINCIPAL (V4.0 CON STREAMING) ---
async def stream_generation_pipeline(final_prompt: str):
    """
    Este es el nuevo orquestador V4.0.
    Usa 'yield' para devolver actualizaciones de estado en tiempo real.
    """
    logger.info("Iniciando pipeline de generación (V4.0 Stream)...")
    global knowledge_base_global
    knowledge_base = knowledge_base_global

    final_workflow_json = None
    final_summary = "Error: El pipeline no generó un resumen."

    try:
        # --- Carga de Modelos ---
        yield "Iniciando modelos de IA... 🧠\n"
        try:
            model = genai.GenerativeModel(GENERATIVE_MODEL)
        except Exception as e:
            logger.error(f"Error inicializando modelo {GENERATIVE_MODEL}: {e}", exc_info=True)
            yield f"ERROR: No se pudo cargar el modelo de IA {GENERATIVE_MODEL}."
            return

        await asyncio.sleep(0.1) # Permite que el 'yield' se envíe

        # --- Paso 1: Investigador ---
        yield "Paso 1: Iniciando Agente Investigador... 🕵️\n"
        investigation_results = agent_investigator(final_prompt, model, knowledge_base)
        if not investigation_results.get("candidate_nodes"):
            yield "ERROR: El Investigador no pudo encontrar nodos candidatos."
            return
        yield f"Investigador encontró {len(investigation_results.get('candidate_nodes', []))} nodos y {len(investigation_results.get('case_studies', []))} casos.\n"
        await asyncio.sleep(0.1)

        # --- INICIO DE CAMBIOS ---
        
        # --- Paso 2: Arquitecto ---
        yield "Paso 2: Iniciando Agente Arquitecto... 🏛️\n"
        # 1. 'architect_output' es ahora el objeto JSON completo
        architect_output = agent_architect(investigation_results, final_prompt, knowledge_base, model)
        if not architect_output:
            yield "ERROR: El Arquitecto no pudo generar un plan."
            return
        
        # 2. Desempaquetamos el plan y las pruebas
        logical_plan = architect_output.get("logical_plan")
        validation_tests = architect_output.get("validation_tests", []) # Lista de pruebas
        
        yield "Arquitecto generó el plan lógico y las pruebas de validación.\n"
        await asyncio.sleep(0.1)

        # --- Paso 3: Builder ---
        # 3. Solo pasamos el 'logical_plan'
        nodes_template, connections = build_nodes_from_plan(logical_plan, knowledge_base)
        if not nodes_template:
            yield "ERROR: El Builder no pudo construir nodos del plan."
            return
        yield f"Builder construyó el esqueleto de {len(nodes_template)} nodos.\n"
        await asyncio.sleep(0.1)

        # --- Paso 4: Configurador ---
        yield "Paso 4: Iniciando Agente Configurador... 🛠️\n"
        nodes_with_params = agent_parameter_configurator(nodes_template, final_prompt, investigation_results, model, knowledge_base)
        yield "Configurador rellenó los parámetros.\n"
        await asyncio.sleep(0.1)

        # --- ¡NUEVO PASO 5: VALIDACIÓN! ---
        yield "Paso 5: Validando lógica interna... 🔍\n"
        validation_result = agent_validator(nodes_with_params, validation_tests)
        
        if validation_result["status"] == "failed":
            # Si falla, detenemos el pipeline y reportamos el error
            errors_str = "\n".join(validation_result["errors"])
            logger.error(f"PIPELINE DETENIDO. Fallo de validación: {errors_str}")
            yield f"ERROR: El plan falló la autocorrección. Errores:\n{errors_str}"
            return # ¡Detenemos todo!
            
        yield "Validación interna superada. El flujo es lógicamente correcto.\n"
        await asyncio.sleep(0.1)
        # --- FIN DEL NUEVO PASO ---

        # --- Paso 6: Redactor Técnico (era el Paso 5) ---
        yield "Paso 6: Iniciando Agente Redactor Técnico... 📝\n"
        nodes_with_instructions = agent_technical_writer(nodes_with_params, final_prompt, model)
        yield "Redactor escribió las notas de ayuda.\n"
        await asyncio.sleep(0.1)

        # --- Paso 7: Ensamblador (era el Paso 6) ---
        yield "Paso 7: Ensamblando workflow final... 🏗️\n"
        final_workflow_str = final_assembler(nodes_with_instructions, connections, final_prompt)
        final_summary = "Workflow generado. Revisa las notas para pasos finales." # Resumen simple

        # --- FIN DE CAMBIOS ---

        final_workflow_json = json.loads(final_workflow_str)

        logger.info("PIPELINE DE GENERACIÓN (V4.0 Stream) COMPLETADO.")

    except Exception as e:
        logger.error(f"Error crítico en V4.0 Stream Pipeline: {e}", exc_info=True)
        yield f"ERROR: Ocurrió un fallo general en el pipeline: {e}"
        return

    # El ÚLTIMO yield debe ser el objeto JSON final, empaquetado como string
    final_output_object = {
        "workflow_json": final_workflow_json,
        "executive_summary": final_summary
    }
    yield json.dumps(final_output_object)


# --- CÓDIGO DE LA API (FastAPI) ---
app = FastAPI(title="Nexus Automator API")

# --- Modelos Pydantic ---
class WorkflowRequest(BaseModel): user_prompt: str
class InterviewRequest(BaseModel): original_prompt: str; questions: list[str] = []; answers: list[str] = []

# --- Evento de Inicio ---
# --- Evento de Inicio ---
@app.on_event("startup")
async def startup_event():
    # ¡Añadimos 'global' para modificar las variables de fuera!
    global chroma_client, kb_collection, exp_collection
    
    logger.info("Evento de inicio: Conectando a ChromaDB...")
    
    # --- AHORA PEGA EL BLOQUE CORTADO AQUÍ ---
    try:
        CHROMA_HOST = os.getenv("CHROMA_SERVER_HOST")
        if not CHROMA_HOST:
            raise ValueError("¡ERROR CRÍTICO! CHROMA_SERVER_HOST no está configurada.")

        logger.info(f"Conectando a ChromaDB Server en: {CHROMA_HOST}")

        # Conexión como cliente HTTP...
        chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=8000)
        embedding_fn_connect = GeminiEmbeddingFunction() 

        chroma_client.heartbeat() # Prueba de conexión

        kb_collection = chroma_client.get_collection(ENCYCLOPEDIA_COLLECTION, embedding_function=embedding_fn_connect)
        exp_collection = chroma_client.get_collection(EXPERIENCE_COLLECTION, embedding_function=embedding_fn_connect)
        logger.info(f"✅ Conectado a colecciones ChromaDB existentes en Servidor: {ENCYCLOPEDIA_COLLECTION} ({kb_collection.count()} docs), {EXPERIENCE_COLLECTION} ({exp_collection.count()} docs)")

    except Exception as e:
        logger.error(f"!!!!!!!!!! ERROR CRÍTICO CONECTANDO A CHROMADB (Servidor) !!!!!!!!!!", exc_info=True)
        chroma_client = None
        kb_collection = None
        exp_collection = None
    # --- FIN DEL BLOQUE PEGADO ---
    
    logger.info("Aplicación FastAPI iniciada y lista.") # Este log ya estaba
# ¡NUEVO ENDPOINT V4.0!
@app.post("/interview/")
async def handle_interview(request: InterviewRequest):
    """
    Maneja la lógica de la entrevista con el Agente Entrevistador.
    """
    logger.info(f"Petición recibida en /interview/ para: '{request.original_prompt[:50]}...'")
    try:
        model = genai.GenerativeModel(GENERATIVE_MODEL) 
        response_data = agent_interviewer(
            request.original_prompt, 
            request.questions, 
            request.answers, 
            model
        )
        return response_data
    except Exception as e:
        logger.error(f"Error fatal en /interview/: {e}", exc_info=True)
        return {"status": "clarified", "briefing": f"Error: {e}"}

# --- (Aquí es donde empieza el @app.post("/create-workflow-streaming/") que ya tenías) ---

@app.post("/create-workflow-streaming/")
async def handle_create_workflow_streaming(request: WorkflowRequest):
    logger.info(f"Petición V4.0 recibida en /create-workflow-streaming/ para: '{request.user_prompt[:50]}...'")
    return StreamingResponse(
        stream_generation_pipeline(request.user_prompt), 
        media_type="text/plain" # Enviamos línea por línea
    )

@app.get("/healthz")
def healthz():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}

@app.get("/readinessz")
def readinessz():
    global knowledge_base_global, chroma_client, kb_collection, exp_collection
    kb_ready = bool(knowledge_base_global)
    chroma_connected = False
    collections_ready = False
    if chroma_client:
        try:
            chroma_client.heartbeat() # Check connection
            chroma_connected = True
            if kb_collection and exp_collection:
                 kb_count = kb_collection.count() # Check collections access
                 exp_count = exp_collection.count()
                 collections_ready = True
                 logger.debug(f"Readiness check: KB count={kb_count}, EXP count={exp_count}")
            else: logger.warning("Readiness check: Colecciones no inicializadas globalmente.")
        except Exception as e:
            logger.warning(f"Readiness check ChromaDB falló: {e}")
            chroma_connected = False
            collections_ready = False
    is_ready = kb_ready and chroma_connected and collections_ready
    return {"kb_loaded": kb_ready, "chroma_connected": chroma_connected, "chroma_collections_accessible": collections_ready, "model_configured": GENERATIVE_MODEL, "ready": is_ready}

@app.get("/")
def read_root():
    return {"status": "Nexus Automator API is running"}

@app.get("/debug-chroma")
async def debug_chroma():
    """
    Endpoint de prueba para inspeccionar 
    directamente los vectores en ChromaDB.
    """
    global kb_collection
    if not kb_collection:
        return {"error": "kb_collection no está inicializada."}

    try:
        # Obtenemos los 5 primeros items Y sus embeddings
        data = kb_collection.get(
            limit=5,
            include=["embeddings", "documents"] 
        )
        return data
    except Exception as e:
        return {"error": f"Error consultando Chroma: {str(e)}"}

        # --- ENDPOINT DE DEBUG DE API ---


@app.get("/debug-api")
async def debug_api():
    """
    Endpoint de prueba para verificar si la GOOGLE_API_KEY
    del backend es válida y puede crear embeddings.
    """
    try:
        logger.info("Iniciando prueba de API en /debug-api...")
        # Intentamos crear un embedding, la misma operación que falla en el Agente
        test_embedding = genai.embed_content(
            model=EMBEDDING_MODEL, 
            content="Esta es una prueba de API", 
            task_type="retrieval_query"
        )

        vector = test_embedding.get('embedding')
        if vector and len(vector) > 10:
            logger.info("Prueba de API exitosa.")
            return {
                "status": "ÉXITO", 
                "message": "La API Key es válida y puede generar embeddings.",
                "vector_preview": vector[:5] 
            }
        else:
            logger.error("La API devolvió una respuesta vacía o inesperada.")
            return {"status": "FALLO", "message": "La API devolvió una respuesta vacía."}

    except Exception as e:
        # Si la clave es inválida, aquí es donde caerá
        logger.error(f"¡FALLO DE API! Error en /debug-api: {str(e)}", exc_info=True)
        return {
            "status": "¡¡FALLO CRÍTICO!!", 
            "message": "La llamada a la API de Google falló.",
            "error": str(e)
        }
