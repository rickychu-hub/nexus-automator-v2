import requests
import json
import logging
import os
import time

# Configuración básica
N8N_BASE_URL = os.getenv("N8N_BASE_URL")
N8N_API_KEY = os.getenv("N8N_API_KEY")

logger = logging.getLogger(__name__)

class N8nDeployer:
    def __init__(self):
        self.headers = {
            "X-N8N-API-KEY": N8N_API_KEY,
            "Content-Type": "application/json"
        }

    def deploy_workflow(self, workflow_json: dict):
        """
        Despliega el workflow en n8n via API.
        LIMPIA propiedades no válidas y corrige estructuras de nodos.
        """
        if not N8N_BASE_URL or not N8N_API_KEY:
            raise Exception("Faltan credenciales N8N_BASE_URL o N8N_API_KEY")

        # 1. SANITIZACIÓN GLOBAL (Nivel Root)
        payload = workflow_json.copy()
        keys_to_remove = ['meta', 'id', 'deployment', 'active'] 
        
        for key in keys_to_remove:
            if key in payload:
                del payload[key]

        # 2. SANITIZACIÓN DE NODOS (Nivel Profundo)
        # Recorremos los nodos para corregir el error de "credentials must be object"
        if "nodes" in payload and isinstance(payload["nodes"], list):
            for node in payload["nodes"]:
                # CORRECCIÓN CRÍTICA:
                # Si 'credentials' es una lista (alucinación de la IA), la borramos.
                # n8n espera un objeto, y como no tenemos IDs reales, mejor enviarlo limpio.
                if "credentials" in node:
                    if isinstance(node["credentials"], list):
                        logger.warning(f"🧹 Limpiando credenciales inválidas (lista) en nodo: {node.get('name')}")
                        del node["credentials"]
                    elif node["credentials"] is None:
                        del node["credentials"]

        # 3. CREAR WORKFLOW (POST)
        create_url = f"{N8N_BASE_URL}/api/v1/workflows"
        logger.info(f"🚀 Enviando workflow a n8n: {create_url}")
        
        response = requests.post(create_url, headers=self.headers, json=payload, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"🔥 Error comunicando con n8n API: {response.text}")
            raise Exception(f"Fallo en despliegue n8n: {response.text}")

        # Datos del workflow creado
        wf_data = response.json()
        wf_id = wf_data.get("id")
        wf_name = wf_data.get("name")

        logger.info(f"✅ Workflow creado. ID: {wf_id}")

        # 4. DOUBLE-TAP PROTOCOL (Activar)
        time.sleep(1)
        
        activate_url = f"{N8N_BASE_URL}/api/v1/workflows/{wf_id}/activate"
        act_response = requests.post(activate_url, headers=self.headers, timeout=10)
        
        if act_response.status_code == 200:
            logger.info(f"✅ Workflow {wf_id} activado correctamente.")
        else:
            logger.warning(f"⚠️ No se pudo activar el workflow {wf_id}: {act_response.text}")

        # 5. OBTENER URL DEL WEBHOOK (Introspección simple)
        webhook_url = None
        nodes = wf_data.get("nodes", [])
        for node in nodes:
            if "webhook" in node.get("type", "").lower():
                path = node.get("parameters", {}).get("path", "")
                if path:
                    # Construye la URL pública. Ajusta si usas túneles o dominios custom.
                    webhook_url = f"{N8N_BASE_URL}/webhook/{path}"
                    break
        
        dashboard_url = f"{N8N_BASE_URL}/workflow/{wf_id}"

        return {
            "status": "deployed",
            "id": wf_id,
            "name": wf_name,
            "webhook_url": webhook_url,
            "dashboard_url": dashboard_url
        }

n8n_deployer = N8nDeployer()