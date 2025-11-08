import logging
import json
import os
import re
import time
import google.generativeai as genai
import chromadb
from dotenv import load_dotenv, find_dotenv
from chromadb.utils import embedding_functions
import sys # Para evitar errores si se ejecuta fuera de un entorno compatible
import glob


# --- CONFIGURACIÓN ---
# Cargar variables de entorno desde .env en la misma carpeta
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
API_KEY = os.getenv("GOOGLE_API_KEY")

# Modelos y Rutas (igual que en main.py)
EMBEDDING_MODEL = 'models/embedding-001'
ENRICHED_KB_PATH = os.path.join(os.path.dirname(__file__), "knowledge_base_final_CURATED.json")
WORKFLOW_SOURCE_DIR = os.path.join(os.path.dirname(__file__), "workflow_source_jsons")
CHROMA_DB_PATH = os.getenv("CHROMA_PERSIST_PATH", "/data/chroma_db_v2")
DATASET_PATH = os.path.join(os.path.dirname(__file__), "workflow_cases_dataset.json") # Ruta al dataset de experiencia

ENCYCLOPEDIA_COLLECTION = 'n8n_nodes_final_v5'
EXPERIENCE_COLLECTION = 'n8n_workflow_cases_v1'

# --- Inicialización Logging ---
# Configurar para que los logs se vean bien en la consola de Render Shell
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Configuración de Google Generative AI ---
if API_KEY:
    try:
        genai.configure(api_key=API_KEY)
        logger.info("Google Generative AI configurado.")
    except Exception as e:
        logger.error(f"Error configurando Google Generative AI: {e}", exc_info=True)
        sys.exit("Error crítico: Fallo al configurar la API de Google.") # Salir si no hay API
else:
    logger.error("¡¡¡ERROR CRÍTICO!!! GOOGLE_API_KEY no encontrada.")
    sys.exit("Error crítico: GOOGLE_API_KEY no definida en .env.")

# --- Función de Embeddings basada en Gemini (Optimizada) ---
class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self):
        super().__init__()
    def __call__(self, texts):
        embeddings = []
        batch_size = 100
        total_batches = (len(texts) + batch_size - 1) // batch_size
        logger.info(f"Iniciando generación de embeddings para {len(texts)} textos en {total_batches} lotes...")
        for i in range(0, len(texts), batch_size):
            current_batch_num = i // batch_size + 1
            batch_texts = texts[i:i + batch_size]
            processed_batch = []
            for t_idx, t in enumerate(batch_texts):
                max_bytes = 30000
                encoded_text = t.encode("utf-8")
                if len(encoded_text) > max_bytes:
                    logger.warning(f"Texto truncado (lote {current_batch_num}, item {t_idx}). Original: {len(encoded_text)} bytes.")
                    decoded_text = encoded_text.decode("utf-8", errors='ignore')
                    truncated_text = decoded_text[:int(max_bytes * 0.9)]
                    processed_batch.append(truncated_text)
                else:
                    processed_batch.append(t)
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL, content=processed_batch, task_type="retrieval_document"
                )
                embeddings.extend(result['embedding'])
                logger.info(f"Embeddings generados para lote {current_batch_num}/{total_batches}")
            except Exception as e:
                logger.error(f"Error embeddings (lote {current_batch_num}): {e}. Null vectors.", exc_info=False)
                # Asegurar dimensión correcta (ej: 768 para embedding-001)
                embeddings.extend([[0.0] * 768] * len(batch_texts))
            # Pausa MÁS LARGA para proceso de carga manual, evitar errores de cuota
            logger.info(f"Pausa de 5 segundos después del lote {current_batch_num}...")
            time.sleep(5)
        logger.info(f"✅ Embeddings procesados: {len(embeddings)}")
        return embeddings

# --- Lógica Principal de Carga ---
def load_data_to_chroma():
    """
    Crea las colecciones de ChromaDB si no existen y las puebla con datos.
    Se ejecuta una sola vez o cuando se necesite refrescar los datos.
    """
    logger.info("--- Iniciando Script de Carga de ChromaDB (V6.2 - S3/Minio) ---")
    try:
        logger.info(f"Conectando a ChromaDB (modo S3 por variables de entorno)...")
        
        # ChromaDB (0.5.x) lee automáticamente las variables de entorno
        # que hemos configurado en el Environment Group de Render.
        chroma_client = chromadb.Client()
        embedding_fn = GeminiEmbeddingFunction()
        
        # El script ahora también creará el "cubo" (bucket) en Minio si no existe
        # (Lo forzamos creando una colección de prueba)
        chroma_client.get_or_create_collection(name="check_bucket_creation")
        
        existing = {c.name for c in chroma_client.list_collections()}
        logger.info(f"Colecciones existentes en S3/Minio: {existing}")
        
        # Definimos el tamaño del lote para añadir a Chroma
        batch_size_add = 2000
        
        # ... (El resto de la función 'load_data_to_chroma' sigue igual) ...
        # --- Cargar Enciclopedia --- (BLOQUE AÑADIDO Y ACTIVADO)
        if ENCYCLOPEDIA_COLLECTION not in existing:
            logger.info(f"Creando y poblando '{ENCYCLOPEDIA_COLLECTION}'...")
            kb_collection = chroma_client.create_collection(
                name=ENCYCLOPEDIA_COLLECTION,
                embedding_function=embedding_fn
            )
            if os.path.exists(ENRICHED_KB_PATH):
                with open(ENRICHED_KB_PATH, "r", encoding="utf-8") as f:
                    kb_data = json.load(f)
                kb_docs = [json.dumps(node, ensure_ascii=False) for node in kb_data.values()]
                kb_ids = list(map(str, kb_data.keys()))
                total_docs = len(kb_ids)
                logger.info(f"Añadiendo {total_docs} documentos a {ENCYCLOPEDIA_COLLECTION} en lotes de {batch_size_add}...")
                for i in range(0, total_docs, batch_size_add):
                    start_idx = i
                    end_idx = min(i + batch_size_add, total_docs)
                    logger.info(f"  Procesando lote {i//batch_size_add + 1}/{(total_docs + batch_size_add - 1)//batch_size_add} (docs {start_idx}-{end_idx-1})...")

                    # Preparamos los datos para este lote
                    batch_ids = kb_ids[start_idx:end_idx]
                    batch_docs = kb_docs[start_idx:end_idx]
                    # ¡¡LA LÍNEA CLAVE!! Creamos las metadatas
                    batch_metadatas = [{"node_id": nid} for nid in batch_ids]

                    kb_collection.add(
                        documents=batch_docs,
                        ids=batch_ids,
                        metadatas=batch_metadatas # <-- ¡AÑADIDO!
                    )
                    logger.info(f"  Lote añadido.")
                    # Pausa adicional entre lotes de escritura a disco
                    time.sleep(2)
                logger.info(f"✅ {ENCYCLOPEDIA_COLLECTION} cargada con {kb_collection.count()} documentos.")
            else:
                logger.error(f"{ENRICHED_KB_PATH} no encontrado.")
        else:
            logger.info(f"Colección '{ENCYCLOPEDIA_COLLECTION}' ya existe. Saltando carga.")


        # --- Cargar Experiencia (¡EL NUEVO CÓDIGO EXPERTO!) ---
        if EXPERIENCE_COLLECTION not in existing:
            logger.info(f"Creando y poblando '{EXPERIENCE_COLLECTION}'...")
            exp_collection = chroma_client.create_collection(
                name=EXPERIENCE_COLLECTION,
                embedding_function=embedding_fn
            )
            
            # Buscamos los 2000+ archivos JSON
            workflow_files = glob.glob(os.path.join(WORKFLOW_SOURCE_DIR, "*.json"))
            logger.info(f"Se encontraron {len(workflow_files)} workflows de experiencia para analizar.")

            exp_docs, exp_ids, exp_metadatas = [], [], []
            doc_counter = 0

            for i, filepath in enumerate(workflow_files):
                if i % 100 == 0:
                    logger.info(f"Procesando workflow {i}/{len(workflow_files)}...")
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        workflow = json.load(f)
                    
                    nodes = workflow.get('nodes', [])
                    connections = workflow.get('connections', {})
                    
                    # Convertimos los nodos en un mapa por ID para fácil acceso
                    node_map = {node.get('id'): node for node in nodes}

                    # 1. Extraer Patrones de Conexión
                    for _start_node_id, conn_data in connections.items():
                        start_node = node_map.get(_start_node_id)
                        if not start_node or not conn_data.get('main'):
                            continue
                        
                        start_type = start_node.get('type')
                        
                        for branch in conn_data['main']:
                            for conn in branch:
                                end_node = node_map.get(conn.get('node'))
                                if not end_node:
                                    continue
                                
                                end_type = end_node.get('type')
                                
                                # ¡La pepita de oro!
                                pattern_text = f"PATRÓN DE CONEXIÓN: Se conecta el nodo {start_type} con el nodo {end_type}."
                                exp_docs.append(pattern_text)
                                exp_ids.append(f"pat_{doc_counter}")
                                exp_metadatas.append({"type": "pattern", "source_file": os.path.basename(filepath)})
                                doc_counter += 1

                    # 2. Extraer Expresiones Mágicas
                    for node in nodes:
                        node_type = node.get('type')
                        parameters = node.get('parameters', {})
                        
                        # Usamos JSON para encontrar expresiones anidadas
                        param_str = json.dumps(parameters)
                        expressions = re.findall(r"({{\s*.*?}})", param_str)
                        
                        for expr in expressions:
                            # ¡La otra pepita de oro!
                            expression_text = f"EXPRESIÓN USADA: En el nodo {node_type}, se usó la expresión: {expr}"
                            exp_docs.append(expression_text)
                            exp_ids.append(f"expr_{doc_counter}")
                            exp_metadatas.append({"type": "expression", "node_type": node_type, "source_file": os.path.basename(filepath)})
                            doc_counter += 1

                except Exception as e:
                    logger.warning(f"Error procesando el archivo {filepath}: {e}")
            
            # Ahora añadimos todos los miles de patrones/expresiones a ChromaDB en lotes
            if exp_docs:
                logger.info(f"Añadiendo {len(exp_docs)} patrones y expresiones a {EXPERIENCE_COLLECTION}...")
                for i in range(0, len(exp_docs), batch_size_add):
                    start_idx = i
                    end_idx = min(i + batch_size_add, len(exp_docs))
                    logger.info(f"  Procesando lote {i//batch_size_add + 1}/{(len(exp_docs) + batch_size_add - 1)//batch_size_add}...")
                    
                    exp_collection.add(
                        documents=exp_docs[start_idx:end_idx],
                        ids=exp_ids[start_idx:end_idx],
                        metadatas=exp_metadatas[start_idx:end_idx]
                    )
                    time.sleep(2) # Pausa
                logger.info(f"✅ {EXPERIENCE_COLLECTION} cargada con {exp_collection.count()} documentos.")
            else:
                logger.warning(f"No se extrajeron patrones ni expresiones de los workflows.")

        else:
            logger.info(f"Colección '{EXPERIENCE_COLLECTION}' ya existe. Saltando carga.")

        logger.info("--- Script de Carga de ChromaDB Finalizado ---")

    except Exception as e:
        logger.error(f"!!!!!!!!!! ERROR CRÍTICO DURANTE LA CARGA DE CHROMADB !!!!!!!!!!", exc_info=True)
        sys.exit("Error crítico: Fallo durante la carga de datos.") # Salir con error

# --- Punto de Entrada del Script ---
if __name__ == "__main__":
    load_data_to_chroma()