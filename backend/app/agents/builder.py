# backend/app/agents/builder.py
import logging
import json
import copy
import time

logger = logging.getLogger(__name__)

# --- 1. BUILDER (Construcción lógica) ---
def build_nodes_from_plan(logical_plan, knowledge_base_memory):
    """
    Convierte el plan lógico del Arquitecto en objetos de nodo reales de n8n.
    """
    logger.info("🏗️ Builder: Construyendo estructura...")
    if not isinstance(logical_plan, list):
        return [], {}

    nodes = []
    connections = {}
    node_counts = {}

    def process_plan_recursive(plan, parent_node_name=None, branch_type=None):
        nonlocal node_counts
        last_node_in_chain = parent_node_name 

        for i, step in enumerate(plan):
            node_id = step.get('nodeId')
            if not node_id: continue

            # Buscado en memoria
            node_template = copy.deepcopy(knowledge_base_memory.get(node_id.lower()))
            
            # Si no existe en KB, usamos un template genérico de seguridad
            if not node_template:
                logger.warning(f"⚠️ Nodo '{node_id}' no en KB. Usando fallback.")
                node_template = {
                    "name": step.get('purpose', 'Node'),
                    "type": node_id,
                    "typeVersion": 1,
                    "position": [0, 0]
                }

            # Configuración
            base_name = node_template.get('name', node_id.split('.')[-1])
            count = node_counts.get(base_name, 0) + 1
            node_counts[base_name] = count
            current_node_name = f"{base_name}_{count}"

            node_template['id'] = f"node_{len(nodes)}_{int(time.time())}" 
            node_template['name'] = current_node_name
            node_template['purpose'] = step.get('purpose', '')
            node_template['parameters'] = step.get('parameters', {})
            
            nodes.append(node_template)

            # Conexiones (Cableado)
            if last_node_in_chain:
                # Lógica simplificada de conexión n8n
                if branch_type is not None and i == 0:
                    # Es el primer nodo de una rama (True/False)
                    branch_index = 0 if branch_type == 'true' else 1
                    connections.setdefault(last_node_in_chain, {"main": [[], []]})
                    # Asegurar que existan suficientes arrays de ramas
                    while len(connections[last_node_in_chain]["main"]) <= branch_index:
                        connections[last_node_in_chain]["main"].append([])
                    
                    connections[last_node_in_chain]["main"][branch_index].append(
                        {"node": current_node_name, "type": "main", "index": 0}
                    )
                else:
                    # Conexión lineal estándar
                    connections.setdefault(last_node_in_chain, {"main": [[]]})
                    if not connections[last_node_in_chain]["main"]:
                         connections[last_node_in_chain]["main"].append([])
                    
                    connections[last_node_in_chain]["main"][0].append(
                        {"node": current_node_name, "type": "main", "index": 0}
                    )
            
            last_node_in_chain = current_node_name

            # Recursividad para ramas (IFs)
            if 'branches' in step and isinstance(step['branches'], dict):
                for branch, sub_plan in step['branches'].items():
                    if isinstance(sub_plan, list):
                        process_plan_recursive(sub_plan, parent_node_name=current_node_name, branch_type=branch)

    process_plan_recursive(logical_plan)
    return nodes, connections

# --- 2. ASSEMBLER (Posicionamiento y Limpieza) ---
def final_assembler(nodes, connections, user_request):
    """
    Calcula posiciones (Layout), limpia y VALIDA el workflow final.
    """
    logger.info("📐 Assembler: Posicionando nodos...")
    
    # Constantes visuales
    X_START, Y_START = 250, 300
    X_SPACING, Y_SPACING = 350, 150
    
    current_x = X_START
    current_y = Y_START
    
    for node in nodes:
        if "position" not in node or node["position"] == [0, 0]:
            node["position"] = [current_x, current_y]
            current_x += X_SPACING
            current_y = Y_START if current_y != Y_START else Y_START + 50

    # Limpieza final de claves
    final_nodes = []
    allowed_keys = ["parameters", "name", "type", "typeVersion", "position", "id", "credentials", "notes"]
    
    for node in nodes:
        clean_node = {k: v for k, v in node.items() if k in allowed_keys}
        final_nodes.append(clean_node)

    # Estructura final preliminar
    workflow_dict = {
        "name": user_request[:60],
        "nodes": final_nodes,
        "connections": connections,
        "active": False,
        "settings": {},
        "meta": {"generated_by": "Nexus OS v4.0"}
    }
    
    # --- 3. SAFETY NET (Validación Final) ---
    # ¡AQUÍ ESTÁ RECUPERADA!
    logger.info("🛡️ Safety Net: Validando integridad estructural...")
    workflow_validated = validar_y_corregir_workflow(workflow_dict)
    
    return json.dumps(workflow_validated, indent=2)

# --- 3. FUNCIÓN DE VALIDACIÓN (Recuperada de V7) ---
def validar_y_corregir_workflow(w: dict) -> dict:
    """
    Repara errores estructurales comunes antes de entregar el JSON.
    """
    if not isinstance(w, dict):
        return w

    # 1. Asegurar IDs únicos
    ids_vistos = set()
    for i, node in enumerate(w.get("nodes", [])):
        node_id = node.get("id")
        if not node_id or node_id in ids_vistos:
            node_id = f"node_{i}_{int(time.time())}"
            node["id"] = node_id
        ids_vistos.add(node_id)

    # 2. Ramas de IF bien definidas en connections
    connections = w.get("connections", {})
    nodes = w.get("nodes", [])
    
    for node in nodes:
        if node.get("type") == "n8n-nodes-base.if":
            node_name = node.get("name")
            if not node_name: continue

            node_conns = connections.setdefault(node_name, {})
            main_conns = node_conns.setdefault("main", [])

            # Asegurar que haya al menos 2 listas (true / false) para el IF
            while len(main_conns) < 2:
                main_conns.append([])

            # Normalizar listas
            for idx in range(2):
                if not isinstance(main_conns[idx], list):
                    main_conns[idx] = []

            node_conns["main"] = main_conns
            connections[node_name] = node_conns

    w["connections"] = connections
    return w