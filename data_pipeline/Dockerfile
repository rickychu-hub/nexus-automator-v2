# data_pipeline/Dockerfile (Versión Segura para Render)
FROM python:3.11-slim
WORKDIR /app

# 1. Copiar SOLO el gestor de dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copiar selectivamente SÓLO los archivos necesarios
# (El .dockerignore ya está evitando que .env se copie)
COPY load_chroma.py .
COPY knowledge_base_final_CURATED.json .
COPY workflow_source_jsons/ ./workflow_source_jsons/

# 3. Ejecutar el script. Las variables de entorno (API_KEY, CHROMA_PERSIST_PATH)
# se inyectarán de forma segura desde el entorno de Render.
CMD ["python", "-u", "load_chroma.py"]