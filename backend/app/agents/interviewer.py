# backend/app/agents/interviewer.py
import logging
import json
import re
import google.generativeai as genai

logger = logging.getLogger(__name__)

def agent_interviewer(original_prompt, questions, answers, model):
    logger.info("🎤 Entrevistador: Analizando requisitos (Modo Estricto V3.0)...")
    
    # Configuración para ser determinista (Cero Creatividad)
    generation_config = genai.types.GenerationConfig(temperature=0.0)

    # --- LÓGICA DE BYPASS (Modo Experto) ---
    # Si el usuario ya ha dado muchos detalles o pide explícitamente generar
    keywords_bypass = ["genera", "json", "no preguntes", "completo", "código", "workflow"]
    is_long_prompt = len(original_prompt) > 300 # Umbral arbitrario de "detalle suficiente"
    has_keywords = any(kw in original_prompt.lower() for kw in keywords_bypass)

    # Construir historial
    history = ""
    if questions and answers:
        history = "\n".join([f"P: {q}\nR: {a}" for q, a in zip(questions, answers) if a])

    # --- CASO 1: CONSOLIDACIÓN (Ya hubo preguntas) ---
    if history:
        logger.info("🎤 Historial detectado. Consolidando briefing.")
        prompt = (
            f"Actúas como un Ingeniero de Sistemas Senior. Tu objetivo es crear un BRIEFING TÉCNICO.\n"
            f"Consolida la petición original y las respuestas del usuario.\n"
            f"**Original:** {original_prompt}\n"
            f"**Historial:**\n{history}\n\n"
            f"**SALIDA:** Devuelve SOLO el texto del briefing técnico final, limpio y estructurado."
        )
        try:
            res = model.generate_content(prompt, generation_config=generation_config)
            return {"status": "clarified", "briefing": res.text.strip()}
        except Exception as e:
            return {"status": "clarified", "briefing": original_prompt + "\n" + history}

    # --- CASO 2: ANÁLISIS INICIAL (¿Necesito preguntar?) ---
    else:
        # Si parece un prompt experto, intentamos saltar la entrevista
        prompt_instruction = (
            "Analiza la siguiente petición de automatización. Tu trabajo es decidir si tienes suficiente información para crear un BOCETO inicial.\n\n"
            "**CRITERIOS PARA NO PREGUNTAR (Status: clarified):**\n"
            "1. Si el usuario describe el Trigger y las Acciones principales.\n"
            "2. Si el usuario pide explícitamente 'Genera', 'Crea el JSON' o 'No preguntes'.\n"
            "3. Si la petición es detallada y técnica.\n"
            "4. NO preguntes por credenciales, IDs, nombres de tablas o correos específicos. Eso se configura después.\n\n"
            "**CRITERIOS PARA PREGUNTAR (Status: needs_more_info):**\n"
            "1. Solo si la petición es extremadamente vaga (ej: 'Quiero automatizar mis ventas').\n\n"
            f"**PETICIÓN:** \"{original_prompt}\"\n\n"
            "**FORMATO DE SALIDA JSON OBLIGATORIO:**\n"
            "```json\n"
            "{ \"status\": \"clarified\", \"briefing\": \"Resumen técnico...\" }\n"
            "// O SI ES MUY VAGO:\n"
            "{ \"status\": \"needs_more_info\", \"questions\": [\"Pregunta clave 1\"] }\n"
            "```"
        )

        try:
            res = model.generate_content(prompt_instruction, generation_config=generation_config)
            text_response = res.text.strip()
            
            # Limpieza de JSON
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text_response, re.DOTALL)
            if not json_match:
                json_match = re.search(r'(\{.*\})', text_response, re.DOTALL)

            if json_match:
                result = json.loads(json_match.group(1))
                logger.info(f"🎤 Decisión del Entrevistador: {result.get('status')}")
                return result
            else:
                # Fallback seguro: Si la IA falla al decidir, asumimos que está claro y avanzamos.
                logger.warning("⚠️ Entrevistador no devolvió JSON. Asumiendo 'clarified' por defecto.")
                return {"status": "clarified", "briefing": original_prompt}

        except Exception as e:
            logger.error(f"Error en Entrevistador: {e}")
            return {"status": "clarified", "briefing": original_prompt}