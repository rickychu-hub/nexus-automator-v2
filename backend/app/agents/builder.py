# backend/app/agents/builder.py
import logging
import json
import copy
import time

logger = logging.getLogger(__name__)

# --- 1. BUILDER (Sin cambios) ---
def build_nodes_from_plan(logical_plan, knowledge_base_memory):
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

            node_template = copy.deepcopy(knowledge_base_memory.get(node_id.lower()))
            if not node_template:
                node_template = {"name": step.get('purpose', 'Node'), "type": node_id, "typeVersion": 1, "position": [0, 0]}

            base_name = node_template.get('name', node_id.split('.')[-1])
            count = node_counts.get(base_name, 0) + 1
            node_counts[base_name] = count
            current_node_name = f"{base_name}_{count}"

            node_template['id'] = f"node_{len(nodes)}_{int(time.time())}" 
            node_template['name'] = current_node_name
            node_template['purpose'] = step.get('purpose', '')
            node_template['parameters'] = step.get('parameters', {})
            
            nodes.append(node_template)

            if last_node_in_chain:
                if branch_type is not None and i == 0:
                    branch_index = 0 if branch_type == 'true' else 1
                    connections.setdefault(last_node_in_chain, {"main": [[], []]})
                    while len(connections[last_node_in_chain]["main"]) <= branch_index:
                        connections[last_node_in_chain]["main"].append([])
                    connections[last_node_in_chain]["main"][branch_index].append({"node": current_node_name, "type": "main", "index": 0})
                else:
                    connections.setdefault(last_node_in_chain, {"main": [[]]})
                    if not connections[last_node_in_chain]["main"]: connections[last_node_in_chain]["main"].append([])
                    connections[last_node_in_chain]["main"][0].append({"node": current_node_name, "type": "main", "index": 0})
            
            last_node_in_chain = current_node_name

            if 'branches' in step and isinstance(step['branches'], dict):
                for branch, sub_plan in step['branches'].items():
                    if isinstance(sub_plan, list):
                        process_plan_recursive(sub_plan, parent_node_name=current_node_name, branch_type=branch)

    process_plan_recursive(logical_plan)
    return nodes, connections

# --- 2. ASSEMBLER DIRECCIONAL (El arreglo visual) ---
def final_assembler(nodes, connections, user_request):
    logger.info("📐 Assembler: Aplicando Layout Direccional...")

    node_map = {n["name"]: n for n in nodes}
    
    X_GAP = 450      
    Y_BRANCH_GAP = 450 # Aumentado un poco más por seguridad
    NOTE_OFFSET = 300 
    
    visited = set()

    # Añadimos un parámetro 'vertical_direction': -1 (Arriba), 1 (Abajo)
    def position_recursive(node_name, x, y, vertical_direction=-1):
        if node_name in visited or node_name not in node_map:
            return
        
        visited.add(node_name)
        node = node_map[node_name]
        
        if "stickyNote" not in node.get("type", ""):
            node["position"] = [x, y]
            # Guardamos la preferencia de dirección en el nodo temporalmente
            node["_note_dir"] = vertical_direction

        if node_name in connections:
            main_conns = connections[node_name].get("main", [])
            
            # Caso IF (Ramas)
            if len(main_conns) > 1:
                # Rama True (Arriba) -> Mantenemos dirección -1 (Notas arriba)
                if main_conns[0]:
                    next_node = main_conns[0][0]["node"]
                    position_recursive(next_node, x + X_GAP, y - (Y_BRANCH_GAP / 2), -1)
                
                # Rama False (Abajo) -> CAMBIAMOS dirección a 1 (Notas abajo)
                if main_conns[1]:
                    next_node = main_conns[1][0]["node"]
                    position_recursive(next_node, x + X_GAP, y + (Y_BRANCH_GAP / 2), 1)
            
            # Caso Lineal (Hereda la dirección del padre)
            elif len(main_conns) == 1 and main_conns[0]:
                next_node = main_conns[0][0]["node"]
                position_recursive(next_node, x + X_GAP, y, vertical_direction)

    # Buscar inicio
    targets = set()
    for conns in connections.values():
        for branch in conns.get("main", []):
            for item in branch: targets.add(item["node"])
    
    start_nodes = [n["name"] for n in nodes if n["name"] not in targets and "stickyNote" not in n["type"]]
    
    if start_nodes:
        # El tronco principal tiene dirección -1 (Notas arriba por defecto)
        position_recursive(start_nodes[0], 200, 600, -1)
    
    # POSICIONAR NOTAS (Usando la dirección guardada)
    for note in nodes:
        if note.get("type") == "n8n-nodes-base.stickyNote":
            note_id = note.get("id", "")
            target_id = note_id.replace("note_", "")
            target_node = next((n for n in nodes if n["id"] == target_id), None)
            
            if target_node and "position" in target_node:
                tx, ty = target_node["position"]
                # Leemos la dirección preferida (-1 o 1), por defecto Arriba (-1)
                direction = target_node.get("_note_dir", -1)
                
                # Calculamos posición:
                # Si direction es -1 (Arriba): ty - 300
                # Si direction es  1 (Abajo):  ty + 150 (Un poco menos offset porque el nodo tiene altura)
                
                offset_pixels = NOTE_OFFSET if direction == -1 else (NOTE_OFFSET * 0.6)
                final_y = ty + (direction * offset_pixels)
                
                # Ajuste extra: Si va abajo, sumamos la altura del nodo aprox (100px)
                if direction == 1:
                    final_y += 100

                note["position"] = [tx, final_y]
            else:
                note["position"] = [0, -600]

    # Limpieza final (Eliminamos la marca temporal _note_dir)
    final_nodes = []
    allowed_keys = ["parameters", "name", "type", "typeVersion", "position", "id", "credentials", "notes"]
    for node in nodes:
        clean_node = {k: v for k, v in node.items() if k in allowed_keys}
        final_nodes.append(clean_node)

    workflow_dict = {
        "name": user_request[:60],
        "nodes": final_nodes,
        "connections": connections,
        "active": False,
        "settings": {},
        "meta": {"generated_by": "Nexus OS v4.7"}
    }
    
    return json.dumps(validar_y_corregir_workflow(workflow_dict), indent=2)

def validar_y_corregir_workflow(w: dict) -> dict:
    if not isinstance(w, dict): return w
    for i, node in enumerate(w.get("nodes", [])):
        if not node.get("id"): node["id"] = f"node_{i}_{int(time.time())}"
    connections = w.get("connections", {})
    nodes = w.get("nodes", [])
    for node in nodes:
        if node.get("type") == "n8n-nodes-base.if":
            name = node.get("name")
            if name:
                conns = connections.setdefault(name, {}).setdefault("main", [])
                while len(conns) < 2: conns.append([])
    w["connections"] = connections
    return w