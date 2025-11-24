# backend/app/agents/investigator.py
import logging
import json
import re
import google.generativeai as genai
from app.core.config import settings
from app.services.chroma_service import get_collections

logger = logging.getLogger(__name__)

def agent_investigator(user_request, model, knowledge_base_memory):
    """
    Agente V2.2 Refactorizado:
    1. Valida nodos contra la KB en memoria.
    2. Busca patrones de experiencia en ChromaDB usando 'metadatas'.
    """
    logger.info("🕵️ Iniciando Agente Investigador (Refactorizado)...")
    kb_collection, exp_collection = get_collections()

    # --- FASE 1: Pensar (Validación contra JSON en memoria) ---
    all_node_names = list(knowledge_base_memory.keys())
    
    prompt = (
        f"Actúas como un Analista de n8n experto. Analiza el briefing y compáralo con la lista de nodos disponibles.\n"
        f"**Briefing:** \"{user_request}\"\n"
        f"**Nodos Disponibles:** {json.dumps(all_node_names[:500], indent=2)}... (lista truncada)\n" # Truncamos para no saturar prompt si es enorme
        f"Tu tarea: Identifica los `nodeId`s EXACTOS necesarios. Si es una app genérica, busca su nodo.\n"
        f"Devuelve JSON: {{ \"required_nodes\": [\"n8n-nodes-base.googleSheets\", ...] }}"
    )

    validated_node_ids = []
    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_response, re.DOTALL)
        if json_match:
            result_json = json.loads(json_match.group(1))
            validated_node_ids = result_json.get("required_nodes", [])
            logger.info(f"✅ Nodos identificados: {validated_node_ids}")
        else:
            logger.warning("⚠️ Investigador no devolvió JSON válido. Usando lista vacía.")
    except Exception as e:
        logger.error(f"❌ Error en Fase Pensar: {e}")

    # --- FASE 2: Buscar Experiencia (RAG sobre Patrones) ---
    case_studies = []
    if exp_collection:
        try:
            # Generamos embedding de consulta (Query)
            # IMPORTANTE: Usamos task_type="retrieval_query" porque load_chroma usó "retrieval_document"
            query_embedding = genai.embed_content(
                model=settings.EMBEDDING_MODEL,
                content=f"Workflow con nodos: {', '.join(validated_node_ids)}. Objetivo: {user_request}",
                task_type="retrieval_query"
            )['embedding']

            # Buscamos patrones similares
            results = exp_collection.query(
                query_embeddings=[query_embedding],
                n_results=5,
                include=['documents', 'metadatas']
            )
            
            if results and results['documents']:
                case_studies = results['documents'][0]
                logger.info(f"📚 Encontrados {len(case_studies)} patrones de experiencia.")
        except Exception as e:
            logger.error(f"❌ Error buscando experiencia: {e}")
    else:
        logger.warning("⚠️ Colección de Experiencia no disponible.")

    return {"candidate_nodes": validated_node_ids, "case_studies": case_studies}