# backend/app/core/config.py
import os
from dotenv import load_dotenv

# Cargar .env desde la raíz del backend (un nivel arriba de app/)
base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(base_path, ".env"))

class Settings:
    PROJECT_NAME: str = "Nexus Automator API"
    VERSION: str = "4.0.0"
    
    # Google AI
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    GENERATIVE_MODEL: str = "models/gemini-2.5-flash"
    EMBEDDING_MODEL: str = "models/embedding-001"
    
    # Paths
    CHROMA_DB_PATH: str = os.getenv("CHROMA_PERSIST_PATH", "/data/chroma_db_v2")
    # Nota: Este path JSON desaparecerá en el futuro, pero lo mantenemos para la transición
    KB_PATH: str = os.path.join(base_path, "knowledge_base_final_CURATED.json")
    
    # Chroma Server (Para Docker)
    CHROMA_SERVER_HOST: str = os.getenv("CHROMA_SERVER_HOST")
    CHROMA_SERVER_PORT: int = int(os.getenv("CHROMA_SERVER_PORT", 8000))
    
    # Collections
    ENCYCLOPEDIA_COLLECTION: str = 'n8n_nodes_final_v5'
    EXPERIENCE_COLLECTION: str = 'n8n_workflow_cases_v1'

# ¡ESTA ES LA LÍNEA QUE TE FALTA!
settings = Settings()