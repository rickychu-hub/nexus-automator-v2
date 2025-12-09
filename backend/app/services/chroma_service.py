# backend/app/services/chroma_service.py
import chromadb
import logging
import os
import json
import time
import glob
from chromadb.utils import embedding_functions
import google.generativeai as genai
from app.core.config import settings
from chromadb.config import Settings

logger = logging.getLogger(__name__)

# Variables globales para mantener la conexión (Patrón Singleton simple)
_chroma_client = None
_kb_collection = None
_exp_collection = None
_knowledge_base_memory = {}

# Clase de Embeddings (Copiada de tu original)
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __call__(self, texts):
        embeddings = []
        batch_size = 50 # Reducido por seguridad
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            try:
                # Usamos el modelo configurado
                result = genai.embed_content(model=settings.EMBEDDING_MODEL, content=batch_texts, task_type="retrieval_document")
                embeddings.extend(result['embedding'])
            except Exception as e:
                logger.error(f"Error embedding batch: {e}")
                embeddings.extend([[0.0] * 768] * len(batch_texts))
            time.sleep(0.5)
        return embeddings

def init_chroma_client():
    global _chroma_client, _kb_collection, _exp_collection, _knowledge_base_memory
    
    # 1. Cargar JSON en memoria (Legacy)
    try:
        if os.path.exists(settings.KB_PATH):
            with open(settings.KB_PATH, 'r', encoding='utf-8') as f:
                kb_original = json.load(f)
                for k, v in kb_original.items():
                    v['type'] = k 
                    _knowledge_base_memory[k.lower()] = v
            logger.info(f"📚 JSON KB cargado en memoria ({len(_knowledge_base_memory)} items).")
        else:
            logger.warning(f"⚠️ No se encontró JSON KB en {settings.KB_PATH}")
    except Exception as e:
        logger.error(f"Error cargando JSON KB: {e}")

# ... dentro de init_chroma_client ...

    # 2. Conectar a Chroma Server
    # 2. Conectar a Chroma Server
    try:
        # Recuperar Host y Port de la configuración (o env vars)
        # Prioridad: Variable de entorno > Configuración > Default Localhost
        host = settings.CHROMA_SERVER_HOST or os.getenv("CHROMA_SERVER_HOST") or "localhost"
        port = settings.CHROMA_SERVER_PORT or os.getenv("CHROMA_SERVER_PORT") or 8000
        
        # Limpieza del Host (por si viene con http/https)
        clean_host = host.replace("http://", "").replace("https://", "").split(":")[0]
        
        # Validación básica de puerto
        try:
            clean_port = int(port)
        except ValueError:
            logger.warning(f"Puerto inválido '{port}', usando 8000 por defecto.")
            clean_port = 8000

        logger.info(f"🔌 Conectando a Chroma en: {clean_host}:{clean_port}")

        # 3. Conexión Dinámica
        global _chroma_client
        _chroma_client = chromadb.HttpClient(host=clean_host, port=clean_port)
    # ---------------------------------------------------
    
        ef = GeminiEmbeddingFunction()
    
    # Ahora sí funcionará esto, porque _chroma_client ya tiene valor
        _kb_collection = _chroma_client.get_collection(settings.ENCYCLOPEDIA_COLLECTION, embedding_function=ef)
        _exp_collection = _chroma_client.get_collection(settings.EXPERIENCE_COLLECTION, embedding_function=ef)
    
        logger.info("✅ Conexión a Chroma establecida.")

    except Exception as e:
        logger.error(f"❌ Error conectando a Chroma: {e}")

def get_collections():
    return _kb_collection, _exp_collection

def get_kb_memory():
    return _knowledge_base_memory

def hydrate_knowledge_base():
    """
    Hydrates ChromaDB with:
    1. Base Knowledge Base (Nodes)
    2. Workflow Examples (ETL + Batching)
    """
    global _chroma_client, _kb_collection, _exp_collection, _knowledge_base_memory
    
    # Aseguramos que el cliente esté inicializado
    if not _chroma_client:
        init_chroma_client()
        
    stats = {"nodes_loaded": 0, "workflows_loaded": 0, "errors": []}
    
    # FIX: Forzamos la obtención/creación de colecciones para evitar errores de inicialización
    try:
        ef = GeminiEmbeddingFunction()
        _kb_collection = _chroma_client.get_or_create_collection(name=settings.ENCYCLOPEDIA_COLLECTION, embedding_function=ef)
        _exp_collection = _chroma_client.get_or_create_collection(name=settings.EXPERIENCE_COLLECTION, embedding_function=ef)
        logger.info("✅ Colecciones inicializadas/verificadas para ingesta.")
    except Exception as e:
        logger.error(f"❌ Error fatal recuperando colecciones: {e}")
        stats["errors"].append(f"Init Error: {str(e)}")
        return stats
    
    # 1. Ingestar Base de Conocimiento (Nodos)
    try:
        # Recargar JSON si es necesario
        if not _knowledge_base_memory and os.path.exists(settings.KB_PATH):
            with open(settings.KB_PATH, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
                _knowledge_base_memory = kb_data
                
        # Preparar documentos
        ids = []
        documents = []
        metadatas = []
        
        for k, v in _knowledge_base_memory.items():
            ids.append(k)
            documents.append(json.dumps(v)) # Serializamos el nodo entero
            metadatas.append({"type": "node_definition", "category": "system"})
            
        if ids:
            _kb_collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            stats["nodes_loaded"] = len(ids)
            logger.info(f"✅ KB Core ingestada: {len(ids)} nodos.")

    except Exception as e:
        logger.error(f"❌ Error ingestando KB Core: {e}")
        stats["errors"].append(str(e))

    # 2. Ingestar Ejemplos Masivos (ETL + Batching)
    try:
        pipeline_path = os.path.join(os.getcwd(), "data_pipeline", "workflow_source_jsons", "*.json")
        json_files = glob.glob(pipeline_path)
        logger.info(f"📁 Encontrados {len(json_files)} archivos de workflow para ingerir.")
        
        batch_size = 50
        current_batch_ids = []
        current_batch_docs = []
        current_batch_metas = []
        
        for file_path in json_files:
            try:
                filename = os.path.basename(file_path)
                with open(file_path, 'r', encoding='utf-8') as f:
                    wf_data = json.load(f)
                    
                # --- ETL: Extracción de Características Clave ---
                # Queremos 'nodes' y 'connections', pero LIMPIOS de posición UI
                clean_nodes = []
                for node in wf_data.get('nodes', []):
                    clean_node = {
                        "name": node.get("name"),
                        "type": node.get("type"),
                        "parameters": node.get("parameters"),
                        # OMITIMOS: position, id, typeVersion (opcional)
                    }
                    clean_nodes.append(clean_node)
                    
                clean_wf = {
                    "nodes": clean_nodes,
                    "connections": wf_data.get("connections", {})
                }
                
                # Crear Documento de Texto para Embedding
                doc_text = json.dumps(clean_wf)
                
                current_batch_ids.append(filename)
                current_batch_docs.append(doc_text)
                current_batch_metas.append({"filename": filename, "source": "manual_ingestion"})
                
                # Procesar Batch
                if len(current_batch_ids) >= batch_size:
                    _exp_collection.upsert(
                        ids=current_batch_ids,
                        documents=current_batch_docs,
                        metadatas=current_batch_metas
                    )
                    stats["workflows_loaded"] += len(current_batch_ids)
                    logger.info(f"   -> Batch procesado ({len(current_batch_ids)} items). Total: {stats['workflows_loaded']}")
                    
                    # Reset Batch
                    current_batch_ids = []
                    current_batch_docs = []
                    current_batch_metas = []
                    
            except Exception as e:
                logger.warning(f"Error procesando archivo {filename}: {e}")
                # No fallamos todo el proceso por un archivo
                
        # Procesar remanentes
        if current_batch_ids:
            _exp_collection.upsert(
                ids=current_batch_ids,
                documents=current_batch_docs,
                metadatas=current_batch_metas
            )
            stats["workflows_loaded"] += len(current_batch_ids)
            logger.info(f"   -> Batch final procesado. Total: {stats['workflows_loaded']}")

    except Exception as e:
        logger.error(f"❌ Error ingestando Workflows: {e}")
        stats["errors"].append(str(e))
        
    return stats