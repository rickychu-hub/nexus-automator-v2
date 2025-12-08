import os
import requests
import time
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class N8nDeployer:
    def __init__(self):
        # Limpiamos la URL base para evitar dobles barras al final
        self.base_url = os.getenv("N8N_BASE_URL", "").rstrip("/")
        self.api_key = os.getenv("N8N_API_KEY")
        
        # Headers estándar para todas las peticiones
        self.headers = {
            "X-N8N-API-KEY": self.api_key,
            "Content-Type": "application/json"
        }

    def _validate_config(self):
        if not self.base_url or not self.api_key:
            raise ValueError("❌ Faltan configuraciones críticas: N8N_BASE_URL o N8N_API_KEY no definidos.")

    def deploy_workflow(self, workflow_json: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ejecuta el protocolo 'Double-Tap' para inyectar y activar el workflow.
        Retorna un resumen con el ID y la URL del Webhook si existe.
        """
        self._validate_config()
        
        # 1. PREPARACIÓN: Forzamos 'active: false' para evitar Race Conditions
        workflow_json['active'] = False
        # Aseguramos que no lleve ID preexistente para que n8n cree uno nuevo limpio
        if 'id' in workflow_json:
            del workflow_json['id']

        try:
            # 2. INYECCIÓN (POST /workflows)
            logger.info("🚀 Enviando workflow a n8n...")
            create_url = f"{self.base_url}/api/v1/workflows"
            response = requests.post(create_url, json=workflow_json, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            result = response.json()
            workflow_id = result.get('id')
            workflow_name = result.get('name')
            
            if not workflow_id:
                raise ValueError("n8n no devolvió un ID de workflow válido.")

            logger.info(f"✅ Workflow creado con ID: {workflow_id}")

            # 3. PAUSA TÁCTICA (Anti-Race Condition)
            # Damos tiempo a la DB de n8n para indexar nodos y triggers
            time.sleep(1.0) 

            # 4. ACTIVACIÓN (POST /workflows/{id}/activate)
            logger.info(f"🔌 Activando workflow {workflow_id}...")
            activate_url = f"{self.base_url}/api/v1/workflows/{workflow_id}/activate"
            act_response = requests.post(activate_url, headers=self.headers, timeout=5)
            act_response.raise_for_status()
            
            logger.info("🟢 Workflow activado correctamente.")

            # 5. RECONSTRUCCIÓN DE URL (Webhook Discovery)
            webhook_url = self._extract_webhook_url(workflow_json)

            return {
                "status": "deployed",
                "id": workflow_id,
                "name": workflow_name,
                "webhook_url": webhook_url,
                "dashboard_url": f"{self.base_url}/workflow/{workflow_id}"
            }

        except requests.exceptions.RequestException as e:
            logger.error(f"🔥 Error comunicando con n8n API: {e}")
            if e.response is not None:
                logger.error(f"Respuesta n8n: {e.response.text}")
            raise ConnectionError(f"Fallo en despliegue n8n: {str(e)}")

    def _extract_webhook_url(self, workflow_json: Dict[str, Any]) -> Optional[str]:
        """
        Analiza el JSON para encontrar nodos Webhook y construir la URL pública.
        Prioriza nodos que se llamen 'Webhook' o sean del tipo 'n8n-nodes-base.webhook'.
        """
        nodes = workflow_json.get('nodes', [])
        webhook_node = None

        # Buscamos el nodo webhook
        for node in nodes:
            if node.get('type') == 'n8n-nodes-base.webhook':
                webhook_node = node
                break
        
        if not webhook_node:
            return None

        # Extraemos el path. Si no tiene, n8n usa el ID, pero asumimos que el Builder pone path.
        params = webhook_node.get('parameters', {})
        path = params.get('path')
        
        # Si es un webhook POST o GET
        method = params.get('httpMethod', 'GET')
        
        if path:
            # Construcción estándar de URL de producción de n8n
            return f"{self.base_url}/webhook/{path}"
        
        return f"{self.base_url}/webhook/test-webhook (Path no definido)"

# Instancia global para importar
n8n_deployer = N8nDeployer()