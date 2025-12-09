import logging
import json
import copy
import time
import re  # <--- IMPORTANTE: Necesario para el Humanizador

logger = logging.getLogger(__name__)

# --- 1. BUILDER (Updated v7.1) ---
def build_nodes_from_plan(logical_plan, knowledge_base_memory):
    logger.info("🏗️ Builder v7.1: Construyendo estructura sanitizada...")
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

            # Obtenemos la plantilla base
            node_template = copy.deepcopy(knowledge_base_memory.get(node_id.lower()))
            if not node_template:
                # Fallback si no está en memoria
                node_template = {
                    "name": step.get('purpose', 'Node'), 
                    "type": node_id, 
                    "typeVersion": 1, 
                    "position": [0, 0]
                }

            # --- [MEJORA CRÍTICA] SANITIZACIÓN DE NOMBRES ---
            # 1. Extraer la parte final del ID técnico (ej: 'gmailTrigger')
            raw_suffix = node_id.split('.')[-1]
            
            # 2. Humanizar: 'gmailTrigger' -> 'Gmail Trigger' (Regex CamelCase)
            # Inserta espacio antes de mayúsculas y capitaliza la primera letra
            human_name = re.sub(r'(?<!^)(?=[A-Z])', ' ', raw_suffix).title()
            
            # 3. Limpiar caracteres feos que pueda haber traído la IA
            human_name = human_name.replace("_", " ").strip()
            
            base_name = human_name

            # 4. Contador Inteligente (Estilo n8n nativo)
            # Si es el primero, NO añade número. Si es el segundo, añade " 2" (con espacio)
            count = node_counts.get(base_name, 0) + 1
            node_counts[base_name] = count
            
            if count == 1:
                current_node_name = base_name
            else:
                current_node_name = f"{base_name} {count}"
            
            # --- FIN MEJORA ---

            # Generamos ID único interno (no visible en UI)
            node_template['id'] = f"node_{len(nodes)}_{int(time.time())}" 
            node_template['name'] = current_node_name
            node_template['purpose'] = step.get('purpose', '')
            
            # Mezclar parámetros: Prioridad a lo que dice el plan, fallback a la plantilla
            params = node_template.get('parameters', {})
            params.update(step.get('parameters', {}))
            node_template['parameters'] = params
            
            nodes.append(node_template)

            # Lógica de conexión (Cableado)
            if last_node_in_chain:
                target_index = forced_index if (forced_index is not None and i == 0) else 0
                connections.setdefault(last_node_in_chain, {"main": []})
                main_outputs = connections[last_node_in_chain]["main"]
                
                # Asegurar que existan suficientes salidas
                while len(main_outputs) <= target_index:
                    main_outputs.append([])
                
                main_outputs[target_index].append({"node": current_node_name, "type": "main", "index": 0})
            

            last_node_in_chain = current_node_name

            # --- [PATTERN: DATA CLEANING] INYECTAR CLEAN DATA NODE DESPUÉS DE IA ---
            # Detectamos si es un nodo de IA para limpiar su salida antes de seguir
            is_ai_node = any(x in node_id for x in ["openAi", "langChain", "sentiment", "googleGemini", "anthropic"])
            
            if is_ai_node:
                set_node_name = f"CleanData {base_name}"
                
                # Check uniqueness
                count_set = node_counts.get(set_node_name, 0) + 1
                node_counts[set_node_name] = count_set
                if count_set > 1: set_node_name = f"{set_node_name} {count_set}"
                
                # Creamos el nodo SET (Data Cleaning)
                set_node = {
                    "name": set_node_name,
                    "type": "n8n-nodes-base.set",
                    "typeVersion": 1,
                    "position": [0,0], # El Assembler lo posicionará
                    "id": f"node_{len(nodes)}_{int(time.time())}_clean",
                    "parameters": {
                        "keepOnlySet": False,
                        "values": {
                            "string": [
                                # Mapeo Profundo para OpenAI / AI Agents
                                # Intenta leer estructuras anidadas complejas primero, luego fallbacks planos
                                { "name": "sentiment", "value": "={{ $json.choices[0].message.content.sentimiento || $json.output.sentiment || $json.sentiment || 'neutral' }}" },
                                { "name": "summary", "value": "={{ $json.choices[0].message.content.resumen || $json.output.text || $json.text || $json.content }}" }
                            ]
                        },
                        "options": {}
                    }
                }
                
                nodes.append(set_node)
                
                # Conectar AI Node -> CleanData Node
                connections.setdefault(current_node_name, {"main": []})
                connections[current_node_name]["main"].append([{"node": set_node_name, "type": "main", "index": 0}])
                
                # [CRÍTICO] Actualizar puntero: Los siguientes nodos (ej: If) deben conectarse al SET
                current_node_name = set_node_name
                last_node_in_chain = set_node_name


            # Procesar Ramas (Recursividad)
            if 'branches' in step and isinstance(step['branches'], dict):
                branch_items = list(step['branches'].items())
                # Ordenar 'true' primero para consistencia visual en IFs
                if "true" in step['branches']:
                    branch_items.sort(key=lambda x: 0 if x[0] == "true" else 1)
                
                for idx, (branch_name, sub_plan) in enumerate(branch_items):
                    if isinstance(sub_plan, list):
                        process_plan_recursive(sub_plan, parent_node_name=current_node_name, forced_index=idx)

    process_plan_recursive(logical_plan)
    return nodes, connections

# --- 2. ASSEMBLER (Unchanged) ---
def final_assembler(nodes, connections, user_request):
    logger.info("📐 Assembler: Aplicando Layout...")

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
    
    # Encontrar nodos iniciales (que no son destino de nadie)
    start_nodes = [n["name"] for n in nodes if n["name"] not in targets and "stickyNote" not in n["type"]]
    
    if start_nodes:
        position_recursive(start_nodes[0], 200, 800, 0)
    
    # Posicionar notas adhesivas
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
        "meta": {"generated_by": "Nexus OS v7.1 (Sanitized)"}
    }
    
    return json.dumps(validar_y_reparar_deep(workflow_dict), indent=2)


# --- 3. THE ENFORCER (Deep Repair) ---
def validar_y_reparar_deep(w: dict) -> dict:
    if not isinstance(w, dict): return w
    
    connections = w.get("connections", {})
    nodes = w.get("nodes", [])
    nodes_map = {n['name']: n for n in nodes}

    for i, node in enumerate(nodes):
        if not node.get("id"): node["id"] = f"node_{i}_{int(time.time())}"

    for node_name, conns in connections.items():
        node = nodes_map.get(node_name)
        if not node: continue
        
        # >>> FIX SWITCH NODE <<<
        if node.get('type') == 'n8n-nodes-base.switch':
            params = node.get('parameters', {})
            main_outputs = conns.get('main', [])
            required_outputs = len(main_outputs)

            # 1. FORZAR MODO 'rules'
            if params.get('mode') == 'expression' or 'conditions' in params:
                logger.warning(f"🔧 Switch '{node_name}': Forzando cambio de 'expression' a 'rules'")
                params['mode'] = 'rules'
            
            # 2. ESTRUCTURA CORRECTA
            if 'rules' not in params: params['rules'] = {}
            if 'values' not in params['rules']: params['rules']['values'] = []
            
            current_rules = params['rules']['values']
            
            # 3. REPARAR REGLAS FALTANTES
            if len(current_rules) < required_outputs:
                missing = required_outputs - len(current_rules)
                for _ in range(missing):
                    current_rules.append({
                        "conditions": {
                            "options": { "caseSensitive": True, "leftValue": "", "typeValidation": "strict", "version": 2 },
                            "conditions": [{ "operator": { "type": "string", "operation": "equals" }, "leftValue": "FIX_ME", "rightValue": "" }],
                            "combinator": "and"
                        }
                    })
            
            node['parameters'] = params

    w["connections"] = connections
    return w