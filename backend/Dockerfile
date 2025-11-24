# backend/Dockerfile
FROM python:3.11-slim

# Evitar que Python genere archivos .pyc y buffer de salida
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Instalar dependencias del sistema (opcional, pero recomendado para compilaciones)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 2. Copiar y instalar requisitos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Copiar la Base de Conocimiento (JSON) a la raíz del contenedor
# Esto es importante para que config.py la encuentre en "base_path"
COPY knowledge_base_final_CURATED.json .

# 4. Copiar EL CÓDIGO MODULAR (La carpeta app entera)
COPY app ./app

# 5. Configurar el puerto (Render inyecta PORT, pero definimos 8000 por defecto)
EXPOSE 8000

# 6. Comando de inicio ACTUALIZADO
# Apunta a "app.main:app" en lugar de "main:app"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]