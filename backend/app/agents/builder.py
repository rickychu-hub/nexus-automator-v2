# backend/app/agents/builder.py
import logging
import json
import copy
import time

logger = logging.getLogger(__name__)

# --- 1. BUILDER (Corrección de Índices de Salida) ---
def build_nodes_from_plan(logical_plan, knowledge_base_memory):
    logger.info("🏗️ Builder: Construyendo estructura con Índices Estrictos...")
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

            # --- CONEXIÓN ---
            if last_node_in_chain:
                # Si es el inicio de una rama, usamos el índice forzado (0, 1, 2...)
                # Si es un paso normal dentro de la rama, usamos siempre 0
                target_index = forced_index if (forced_index is not None and i == 0) else 0
                
                connections.setdefault(last_node_in_chain, {"main": []})
                main_outputs = connections[last_node_in_chain]["main"]
                
                # Rellenar con arrays vacíos hasta llegar al índice necesario
                # (Ej: Si target es 2, necesitamos [ [], [], [aquí] ])
                while len(main_outputs) <= target_index:
                    main_outputs.append([])
                
                main_outputs[target_index].append({"node": current_node_name, "type": "main", "index": 0})
            
            last_node_in_chain = current_node_name

            # --- RAMIFICACIÓN ---
            if 'branches' in step and isinstance(step['branches'], dict):
                branch_items = list(step['branches'].items())
                
                # Lógica especial SOLO para IF (true=0, false=1)
                if step.get('nodeId', '').endswith('.if') or "true" in step['branches']:
                    # Aseguramos orden: true primero, false después
                    branch_items.sort(key=lambda x: 0 if x[0] == "true" else 1)
                
                # Para Switch u otros, respetamos el orden natural (0, 1, 2...)
                for idx, (branch_name, sub_plan) in enumerate(branch_items):
                    if isinstance(sub_plan, list):
                        process_plan_recursive(sub_plan, parent_node_name=current_node_name, forced_index=idx)

    process_plan_recursive(logical_plan)
    return nodes, connections

# --- 2. ASSEMBLER (Layout Rascacielos 700px) ---
def final_assembler(nodes, connections, user_request):
    logger.info("📐 Assembler: Aplicando Layout Rascacielos...")

    node_map = {n["name"]: n for n in nodes}
    
    X_GAP = 450
    # ¡AUMENTADO A 700! Para que las notas del medio no toquen a los de arriba
    Y_PER_BRANCH_GAP = 700 
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
            # Contamos ramas reales (no vacías)
            active_branches = [b for b in main_conns if b]
            num_branches = len(active_branches)
            
            # Si hay múltiples salidas (Switch/If)
            if num_branches > 1:
                # Centro lógico
                mid_point = (len(main_conns) - 1) / 2
                
                for idx, branch in enumerate(main_conns):
                    if not branch: continue
                    next_node = branch[0]["node"]
                    
                    # Desviación: -1 (Arriba), 0 (Centro), 1 (Abajo)
                    deviation = idx - mid_point
                    
                    # Nueva Y masiva
                    new_y = y + (deviation * Y_PER_BRANCH_GAP)
                    
                    # Dirección de Notas:
                    # Arriba (-1) para todo lo que esté en el centro o arriba
                    # Abajo (1) SOLO para lo que esté estrictamente abajo
                    new_dir = 1 if deviation > 0 else -1
                    
                    position_recursive(next_node, x + X_GAP, new_y, new_dir)
            
            # Línea recta
            elif len(main_conns) >= 1:
                # Buscamos el primer camino no vacío
                for branch in main_conns:
                    if branch:
                        next_node = branch[0]["node"]
                        position_recursive(next_node, x + X_GAP, y, vertical_direction)
                        break

    # Inicio
    targets = set()
    for conns in connections.values():
        for branch in conns.get("main", []):
            for item in branch: targets.add(item["node"])
    
    start_nodes = [n["name"] for n in nodes if n["name"] not in targets and "stickyNote" not in n["type"]]
    
    if start_nodes:
        # Empezamos muy abajo (Y=1000) para dar margen al techo
        position_recursive(start_nodes[0], 200, 1000, -1) 
    
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
                if direction == 1: final_y += 120 # Un poco más de margen abajo

                note["position"] = [tx, final_y]
            else:
                note["position"] = [0, 0]

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
        "meta": {"generated_by": "Nexus OS v5.2 (Switch Fix)"}
    }
    
    return json.dumps(validar_y_corregir_workflow(workflow_dict), indent=2)

def validar_y_corregir_workflow(w: dict) -> dict:
    if not isinstance(w, dict): return w
    connections = w.get("connections", {})
    # Asegurar que 'main' siempre sea lista de listas
    for key, val in connections.items():
        if "main" in val and isinstance(val["main"], list):
            # Validación extra si fuera necesaria
            pass
    w["connections"] = connections
    return w