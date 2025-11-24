# backend/app/agents/builder.py
import logging
import json
import copy
import time

logger = logging.getLogger(__name__)

# --- 1. BUILDER (Conexiones Estrictas) ---
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

            # --- CONEXIÓN SEGURA ---
            if last_node_in_chain:
                # Si es el primer nodo de una rama, usa el índice forzado. Si no, índice 0.
                target_index = forced_index if (forced_index is not None and i == 0) else 0
                
                connections.setdefault(last_node_in_chain, {"main": []})
                main_outputs = connections[last_node_in_chain]["main"]
                
                # Rellenar huecos con listas vacías para mantener el orden de los índices
                while len(main_outputs) <= target_index:
                    main_outputs.append([])
                
                main_outputs[target_index].append({"node": current_node_name, "type": "main", "index": 0})
            
            last_node_in_chain = current_node_name

            # --- RAMIFICACIÓN (IF/SWITCH) ---
            if 'branches' in step and isinstance(step['branches'], dict):
                branch_items = list(step['branches'].items())
                
                # Ordenar IFs (true=0, false=1)
                if "true" in step['branches']:
                    branch_items.sort(key=lambda x: 0 if x[0] == "true" else 1)
                
                for idx, (branch_name, sub_plan) in enumerate(branch_items):
                    if isinstance(sub_plan, list):
                        process_plan_recursive(sub_plan, parent_node_name=current_node_name, forced_index=idx)

    process_plan_recursive(logical_plan)
    return nodes, connections

# --- 2. ASSEMBLER CON GRAVEDAD (Anti-Colisión) ---
def final_assembler(nodes, connections, user_request):
    logger.info("📐 Assembler: Aplicando Layout con Gravedad...")

    node_map = {n["name"]: n for n in nodes}
    
    X_GAP = 450
    Y_PER_BRANCH_GAP = 600 # Gran espacio vertical
    NOTE_OFFSET = 300
    
    visited = set()

    # vertical_direction: -1 (Norte/Arriba), 1 (Sur/Abajo), 0 (Neutro)
    def position_recursive(node_name, x, y, vertical_direction=0):
        if node_name in visited or node_name not in node_map:
            return
        
        visited.add(node_name)
        node = node_map[node_name]
        
        if "stickyNote" not in node.get("type", ""):
            node["position"] = [x, y]
            node["_note_dir"] = vertical_direction if vertical_direction != 0 else -1

        if node_name in connections:
            main_conns = connections[node_name].get("main", [])
            num_branches = len([b for b in main_conns if b]) # Ramas activas
            
            if num_branches > 1:
                mid_point = (len(main_conns) - 1) / 2
                
                for idx, branch in enumerate(main_conns):
                    if not branch: continue
                    next_node = branch[0]["node"]
                    
                    # Cálculo de desviación base
                    deviation = idx - mid_point
                    
                    # --- LÓGICA DE GRAVEDAD (Corrección de Trayectoria) ---
                    # Si ya estamos en el SUR (1), prohibido ir al NORTE (desviación negativa)
                    # Forzamos que vaya más al SUR o recto.
                    final_deviation = deviation
                    
                    if vertical_direction == 1 and deviation < 0:
                        # Estamos abajo pero la rama quiere subir -> La forzamos a ir recto/abajo levemente
                        final_deviation = 0.5 
                    
                    elif vertical_direction == -1 and deviation > 0:
                        # Estamos arriba pero la rama quiere bajar -> La forzamos a ir recto/arriba levemente
                        final_deviation = -0.5

                    # Nueva Y
                    new_y = y + (final_deviation * Y_PER_BRANCH_GAP)
                    
                    # Nueva Dirección (Heredamos la gravedad fuerte)
                    new_dir = vertical_direction
                    if new_dir == 0: # Si éramos neutros, definimos nueva dirección
                        new_dir = 1 if final_deviation > 0 else -1
                    
                    position_recursive(next_node, x + X_GAP, new_y, new_dir)
            
            elif len(main_conns) >= 1:
                # Búsqueda del primer camino no vacío
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
        position_recursive(start_nodes[0], 200, 800, 0) # Empezamos en Y=800 (Neutro)
    
    # POSICIONAR NOTAS
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

    # Limpieza y Validación Final
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
        "meta": {"generated_by": "Nexus OS v5.5 (Gravity Layout)"}
    }
    
    return json.dumps(validar_y_corregir_workflow(workflow_dict), indent=2)

def validar_y_corregir_workflow(w: dict) -> dict:
    if not isinstance(w, dict): return w
    connections = w.get("connections", {})
    # Aseguramos que 'main' sea lista de listas (formato n8n)
    for key, val in connections.items():
        if "main" in val and isinstance(val["main"], list):
            pass
    w["connections"] = connections
    return w