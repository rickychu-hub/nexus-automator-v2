# backend/app/agents/builder.py
import logging
import json
import copy
import time

logger = logging.getLogger(__name__)

# --- 1. BUILDER (Lógica intacta) ---
def build_nodes_from_plan(logical_plan, knowledge_base_memory):
    logger.info("🏗️ Builder: Construyendo estructura...")
    if not isinstance(logical_plan, list):
        return [], {}

    nodes = []
    connections = {}
    node_counts = {}

    def process_plan_recursive(plan, parent_node_name=None, forced_index=None):
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
                target_index = forced_index if (forced_index is not None and i == 0) else 0
                connections.setdefault(last_node_in_chain, {"main": []})
                main_outputs = connections[last_node_in_chain]["main"]
                while len(main_outputs) <= target_index:
                    main_outputs.append([])
                main_outputs[target_index].append({"node": current_node_name, "type": "main", "index": 0})
            
            last_node_in_chain = current_node_name

            if 'branches' in step and isinstance(step['branches'], dict):
                branch_items = list(step['branches'].items())
                if "true" in step['branches']:
                    branch_items.sort(key=lambda x: 0 if x[0] == "true" else 1)
                for idx, (branch_name, sub_plan) in enumerate(branch_items):
                    if isinstance(sub_plan, list):
                        process_plan_recursive(sub_plan, parent_node_name=current_node_name, forced_index=idx)

    process_plan_recursive(logical_plan)
    return nodes, connections

# --- 2. ASSEMBLER (Con Gravedad y Espacio) ---
def final_assembler(nodes, connections, user_request):
    logger.info("📐 Assembler: Aplicando Layout con Gravedad y Reparación...")

    node_map = {n["name"]: n for n in nodes}
    X_GAP, Y_PER_BRANCH_GAP, NOTE_OFFSET = 450, 600, 300
    visited = set()

    def position_recursive(node_name, x, y, vertical_direction=0):
        if node_name in visited or node_name not in node_map: return
        visited.add(node_name)
        node = node_map[node_name]
        
        if "stickyNote" not in node.get("type", ""):
            node["position"] = [x, y]
            node["_note_dir"] = vertical_direction if vertical_direction != 0 else -1

        if node_name in connections:
            main_conns = connections[node_name].get("main", [])
            num_branches = len([b for b in main_conns if b])
            
            if num_branches > 1:
                mid_point = (len(main_conns) - 1) / 2
                for idx, branch in enumerate(main_conns):
                    if not branch: continue
                    next_node = branch[0]["node"]
                    deviation = idx - mid_point
                    
                    final_deviation = deviation
                    if vertical_direction == 1 and deviation < 0: final_deviation = 0.5 
                    elif vertical_direction == -1 and deviation > 0: final_deviation = -0.5

                    new_y = y + (final_deviation * Y_PER_BRANCH_GAP)
                    new_dir = vertical_direction
                    if new_dir == 0: new_dir = 1 if final_deviation > 0 else -1
                    
                    position_recursive(next_node, x + X_GAP, new_y, new_dir)
            
            elif len(main_conns) >= 1:
                for branch in main_conns:
                    if branch:
                        next_node = branch[0]["node"]
                        position_recursive(next_node, x + X_GAP, y, vertical_direction)
                        break

    targets = set()
    for conns in connections.values():
        for branch in conns.get("main", []):
            for item in branch: targets.add(item["node"])
    
    start_nodes = [n["name"] for n in nodes if n["name"] not in targets and "stickyNote" not in n["type"]]
    if start_nodes:
        position_recursive(start_nodes[0], 200, 800, 0)
    
    for note in nodes:
        if note.get("type") == "n8n-nodes-base.stickyNote":
            note_id = note.get("id", "")
            target_id = note_id.replace("note_", "")
            target_node = next((n for n in nodes if n["id"] == target_id), None)
            if target_node and "position" in target_node:
                tx, ty = target_node["position"]
                direction = target_node.get("_note_dir", -1)
                offset = NOTE_OFFSET if direction == -1 else (NOTE_OFFSET * 0.6)
                final_y = ty + (direction * offset)
                if direction == 1: final_y += 120
                note["position"] = [tx, final_y]
            else:
                note["position"] = [0, -600]

    # Limpieza
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
        "meta": {"generated_by": "Nexus OS v6.0 (Auto-Repair)"}
    }
    
    # ¡AQUÍ LLAMAMOS AL NUEVO REPARADOR!
    return json.dumps(validar_y_corregir_workflow(workflow_dict), indent=2)

# --- 3. SAFETY NET (Reparador de Conexiones Switch) ---
def validar_y_corregir_workflow(w: dict) -> dict:
    if not isinstance(w, dict): return w
    
    connections = w.get("connections", {})
    nodes = w.get("nodes", [])
    nodes_map = {n['name']: n for n in nodes}

    # 1. IDs únicos
    for i, node in enumerate(nodes):
        if not node.get("id"): node["id"] = f"node_{i}_{int(time.time())}"

    # 2. REPARACIÓN DE SWITCHES
    for node_name, conns in connections.items():
        node = nodes_map.get(node_name)
        if not node or node.get('type') != 'n8n-nodes-base.switch': continue

        # Cuántas salidas estamos intentando usar
        main_outputs = conns.get('main', [])
        required_outputs = len(main_outputs)

        # Cuántas reglas tiene configuradas el nodo realmente
        params = node.get('parameters', {})
        # El Switch puede tener 'rules' o 'conditions' dependiendo de la versión/modo
        list_key = 'rules' if 'rules' in params else 'conditions'
        
        # Si no existe la lista, la creamos
        if list_key not in params: 
            list_key = 'rules'
            params[list_key] = []

        current_rules = len(params[list_key])

        # Si usamos más salidas de las que hay reglas, INYECTAMOS reglas fantasma
        # para que n8n pinte las líneas.
        if required_outputs > current_rules:
            missing = required_outputs - current_rules
            logger.warning(f"🔧 Reparando Switch '{node_name}': Faltan {missing} reglas. Inyectando...")
            
            for _ in range(missing):
                # Inyectamos una regla genérica "FIX_ME"
                # Esto habilita el pin de salida visualmente
                if list_key == 'rules':
                    params[list_key].append({
                        "value": "FIX_ME_AI_MISSED_THIS",
                        "operation": "equal" # Ajuste básico
                    })
                else:
                    params[list_key].append({
                        "value1": "FIX_ME",
                        "operation": "equal",
                        "value2": "FIX_ME"
                    })
            
            node['parameters'] = params # Guardamos el cambio

    w["connections"] = connections
    return w