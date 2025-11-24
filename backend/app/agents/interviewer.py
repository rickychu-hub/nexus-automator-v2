# backend/app/agents/interviewer.py
import logging
import json
import re

logger = logging.getLogger(__name__)

def agent_interviewer(original_prompt, questions, answers, model):
    logger.info("🎤 Entrevistador activado...")
    
    # Construir historial
    history = ""
    if questions and answers:
        history = "\n".join([f"P: {q}\nR: {a}" for q, a in zip(questions, answers) if a])

    # Si hay historial, consolidamos (Modo 2)
    if history:
        prompt = (
            f"Consolida la petición original y las respuestas en un briefing técnico final.\n"
            f"Original: {original_prompt}\nHistorial:\n{history}\n"
            f"Devuelve SOLO el texto del briefing."
        )
        try:
            res = model.generate_content(prompt)
            return {"status": "clarified", "briefing": res.text.strip()}
        except Exception as e:
            return {"status": "clarified", "briefing": original_prompt + "\n" + history}

    # Si es nuevo, preguntamos (Modo 1)
    else:
        prompt = (
            f"Analiza: \"{original_prompt}\". Si faltan datos clave (Trigger, Apps, Lógica), genera hasta 2 preguntas.\n"
            f"Formato JSON: {{ \"status\": \"needs_more_info\", \"questions\": [...] }} o {{ \"status\": \"clarified\", \"briefing\": ... }}"
        )
        try:
            res = model.generate_content(prompt)
            match = re.search(r'```json\s*(\{.*?\})\s*```', res.text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return {"status": "clarified", "briefing": original_prompt}
        except Exception:
            return {"status": "clarified", "briefing": original_prompt}