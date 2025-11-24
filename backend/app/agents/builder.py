# backend/app/agents/builder.py
import logging
import json
import copy
import time

logger = logging.getLogger(__name__)

# --- 1. BUILDER (Soporte Multi-Rama) ---
def build_nodes_from_plan(logical_plan, knowledge_base_memory):
    logger.info("🏗️ Builder: Construyendo estructura Multi-Rama...")
    if not isinstance(logical_plan, list):
        return [], {}

    nodes = []
    connections = {}
    node_counts = {}

    # Ahora aceptamos 'forced_index' para saber exactamente por qué salida conectar
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

            # --- LÓGICA DE CONEXIÓN CORREGIDA ---
            if last_node_in_chain:
                # Si es el primer nodo de una sub-rama, usamos el índice forzado que nos pasaron
                if forced_index is not None and i == 0:
                    target_index = forced_index
                else:
                    # Si es una conexión lineal normal, siempre es la salida 0
                    target_index = 0
                
                connections.setdefault(last_node_in_chain, {"main": []})
                main_outputs = connections[last_node_in_chain]["main"]
                
                # Asegurar que existen arrays de salida suficientes
                while len(main_outputs) <= target_index:
                    main_outputs.append([])
                
                # Conectar
                main_outputs[target_index].append({"node": current_node_name, "type": "main", "index": 0})
            
            last_node_in_chain = current_node_name

            # --- RECURSIVIDAD MULTI-RAMA (SWITCH/IF) ---
            if 'branches' in step and isinstance(step['branches'], dict):
                # Iteramos las ramas y asignamos índices numéricos secuenciales (0, 1, 2...)
                # Esto soporta IF (2 ramas) y SWITCH (N ramas)
                branch_items = list(step['branches'].items())
                
                # Truco para IF: Si las claves son "true"/"false", intentamos ordenarlas para que true sea 0
                if "true" in step['branches']:
                    # Reordenar forzosamente: true primero (0), false segundo (1)
                    branch_items.sort(key=lambda x: 0 if x[0] == "true" else 1)

                for idx, (branch_name, sub_plan) in enumerate(branch_items):
                    if isinstance(sub_plan, list):
                        # Pasamos el índice numérico (0, 1, 2) a la recursión
                        process_plan_recursive(sub_plan, parent_node_name=current_node_name, forced_index=idx)

    process_plan_recursive(logical_plan)
    return nodes, connections

# --- 2. ASSEMBLER DIRECCIONAL (Soporte Visual 'El Pulpo') ---
def final_assembler(nodes, connections, user_request):
    logger.info("📐 Assembler: Aplicando Layout Multi-Rama...")

    node_map = {n["name"]: n for n in nodes}
    
    X_GAP = 400
    # Espaciado vertical base. Se multiplicará según el número de ramas.
    Y_BASE_GAP = 200 
    NOTE_OFFSET = 280
    
    visited = set()

    def position_recursive(node_name, x, y, vertical_direction=-1):
        if node_name in visited or node_name not in node_map:
            return
        
        visited.add(node_name)
        node = node_map[node_name]
        
        if "stickyNote" not in node.get("type", ""):
            node["position"] = [x, y]
            node["_note_dir"] = vertical_direction

        if node_name in connections:
            main_conns = connections[node_name].get("main", [])
            num_branches = len(main_conns)

            # Caso Multi-Rama (Switch / If)
            if num_branches > 1:
                # Calculamos el "centro" para distribuir ramas simétricamente
                # Ejemplo 3 ramas: índices 0, 1, 2. 
                # Queremos offsets: -1 (Arriba), 0 (Centro), 1 (Abajo)
                
                mid_point = (num_branches - 1) / 2
                
                for idx, branch in enumerate(main_conns):
                    if not branch: continue
                    next_node = branch[0]["node"]
                    
                    # Factor de desviación vertical
                    # Si idx < mid_point -> Va arriba
                    # Si idx > mid_point -> Va abajo
                    deviation = idx - mid_point
                    
                    # Gap dinámico: cuantas más ramas, más espacio necesitamos
                    dynamic_gap = Y_BASE_GAP * (num_branches - 1) * 0.8
                    
                    # Nueva Y
                    new_y = y + (deviation * dynamic_gap)
                    
                    # Dirección de Notas:
                    # Si la rama va arriba (deviation < 0) -> Notas Arriba (-1)
                    # Si la rama va abajo (deviation > 0) -> Notas Abajo (1)
                    # Si es la del medio (0) -> Alternamos o mandamos arriba (-1)
                    new_dir = 1 if deviation > 0 else -1
                    
                    position_recursive(next_node, x + X_GAP, new_y, new_dir)
            
            # Caso Lineal
            elif len(main_conns) == 1 and main_conns[0]:
                next_node = main_conns[0][0]["node"]
                position_recursive(next_node, x + X_GAP, y, vertical_direction)

    # Inicio
    targets = set()
    for conns in connections.values():
        for branch in conns.get("main", []):
            for item in branch: targets.add(item["node"])
    start_nodes = [n["name"] for n in nodes if n["name"] not in targets and "stickyNote" not in n["type"]]
    
    if start_nodes:
        position_recursive(start_nodes[0], 200, 600, -1)
    
    # Posicionar Notas
    for note in nodes:
        if note.get("type") == "n8n-nodes-base.stickyNote":
            note_id = note.get("id", "")
            target_id = note_id.replace("note_", "")
            target_node = next((n for n in nodes if n["id"] == target_id), None)
            
            if target_node and "position" in target_node:
                tx, ty = target_node["position"]
                direction = target_node.get("_note_dir", -1)
                
                offset_pixels = NOTE_OFFSET if direction == -1 else (NOTE_OFFSET * 0.6)
                final_y = ty + (direction * offset_pixels)
                if direction == 1: final_y += 100

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
        "meta": {"generated_by": "Nexus OS v5.0 (Multi-Branch)"}
    }
    
    return json.dumps(validar_y_corregir_workflow(workflow_dict), indent=2)

def validar_y_corregir_workflow(w: dict) -> dict:
    if not isinstance(w, dict): return w
    for i, node in enumerate(w.get("nodes", [])):
        if not node.get("id"): node["id"] = f"node_{i}_{int(time.time())}"
    # Safety Net para Switch e IF
    connections = w.get("connections", {})
    # (Ya no forzamos 2 ramas estrictas, dejamos que la lógica dinámica mande, 
    # pero aseguramos que 'main' sea una lista de listas)
    for key, val in connections.items():
        if "main" in val and not isinstance(val["main"], list):
            val["main"] = []
    w["connections"] = connections
    return w