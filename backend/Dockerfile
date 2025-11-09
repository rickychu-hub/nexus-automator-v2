FROM python:3.11-slim
WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

# Copia solo los requisitos primero
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia explícitamente solo lo que la app necesita
COPY main.py .
COPY knowledge_base_final_CURATED.json .
# ¡Fíjate que ya no hay 'COPY . .'!

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]