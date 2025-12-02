# backend/app/services/chroma_service.py
import chromadb
import logging
import os
import json
import time
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
    try:
        if not settings.CHROMA_SERVER_HOST:
            raise ValueError("CHROMA_SERVER_HOST no definido.")
            
    # 1. Recuperamos la variable (que vendrá de Render o del .env)
        # En backend usa settings.CHROMA_SERVER_HOST
        # En pipeline usa os.getenv("CHROMA_SERVER_HOST")
        raw_host = settings.CHROMA_SERVER_HOST # (o la variable correspondiente)

        # 2. Limpieza A PRUEBA DE BALAS (Quitamos http:// y puertos)
        clean_host = raw_host.replace("http://", "").replace("https://", "").split(":")[0]
        
        logger.info(f"🔌 Conectando a: {clean_host}")

        # 3. Conexión Dinámica (Ya no está "hardcoded")
        global _chroma_client
        _chroma_client = chromadb.HttpClient(host=clean_host, port=8000)
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