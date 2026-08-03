import re
from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.db.mongo_singleton import MongoSingleton
from huggingface_hub import InferenceClient


class ChatService:
    def __init__(self, scraper_service):
        self.scraper_service = scraper_service
        self.collection = MongoSingleton.get_instance().get_collection()
        self.llm = None

    def answer_question(self, session_id: str, question: str) -> dict[str, Any]:
        relevant_chunks = self.scraper_service.retrieve_relevant_chunks(question, top_k=3)
        debug_payload = {
            "query": question,
            "retrieved_chunks": relevant_chunks,
            "prompt": None,
            "llm_available": True,
            "llm_response": None,
            "llm_model": settings.LLM_MODEL,
        }
        if not relevant_chunks:
            answer = "Todavía no hay contenido indexado en el RAG. Primero guarda una URL para cargar el conocimiento."
            sources = []
        else:
            best_chunk = relevant_chunks[0]
            if self._is_noise_text(best_chunk["text"]):
                answer = "No pude encontrar una respuesta clara con el contenido actual. Intenta otra pregunta o indexa más contenido."
                sources = []
            else:
                prompt = self._build_prompt(question, best_chunk["text"])
                debug_payload["prompt"] = prompt
                llm_answer, llm_available, llm_error = self._generate_response(
                    best_chunk["text"], question
                )
                debug_payload["llm_response"] = llm_answer
                debug_payload["llm_available"] = llm_available
                debug_payload["llm_error"] = llm_error
                answer = llm_answer
                sources = [
                    {"url": item["source_url"], "score": round(item["score"], 3)} for item in relevant_chunks
                ]

        self._persist_history(session_id, question, answer)
        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
            "debug": debug_payload,
        }

    def _is_noise_text(self, text: str) -> bool:
        alpha_chars = sum(1 for ch in text if ch.isalpha())
        if alpha_chars / max(1, len(text)) < 0.45:
            return True
        if len(re.findall(r"[\{\}\[\]<>]", text)) > len(text) * 0.2:
            return True
        return False

    def _compose_answer(self, context: str) -> str:
        cleaned_context = context.strip().replace("\n", " ")
        if len(cleaned_context) > 650:
            cleaned_context = cleaned_context[:650] + "..."
        return f"Según la información indexada, {cleaned_context}"

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            "Eres un asistente en español. Responde de forma breve y específica usando solo el contexto. "
            "No añadas información extra ni asumas datos que no aparecen en el contexto. "
            "Si la respuesta no puede ser inferida del contexto, responde exactamente: No tengo suficiente información."
            f"\nContexto: {context}\nPregunta: {question}\nRespuesta:"
        )

    def _is_invalid_model_output(self, output: str, context: str) -> bool:
        normalized_output = output.strip().lower()
        normalized_context = context.strip().lower()
        if not normalized_output:
            return True
        if normalized_output == normalized_context:
            return True
        if normalized_output.startswith(normalized_context):
            return True
        if "pregunta:" in normalized_output or "respuesta:" in normalized_output:
            return True
        return False

    def _system_prompt(self) -> str:
        return (
            "Eres un asistente en español. Responde solo con la información que "
            "está en el contexto, sin inventar nada. Si no hay una respuesta "
            "clara en el contexto, responde: No tengo suficiente información."
        )

    def _load_llm(self) -> InferenceClient:
        if self.llm is None:
            if not settings.HF_TOKEN:
                raise RuntimeError(
                    "HF_TOKEN no está configurado en .env. Genera un token "
                    "gratis en https://huggingface.co/settings/tokens y "
                    "agrégalo como HF_TOKEN."
                )
            self.llm = InferenceClient(model=settings.LLM_MODEL, token=settings.HF_TOKEN)
        return self.llm

    def _generate_response(self, context: str, question: str) -> tuple[str, bool, str | None]:
        try:
            client = self._load_llm()
            completion = client.chat_completion(
                messages=[
                    {"role": "system", "content": self._system_prompt()},
                    {
                        "role": "user",
                        "content": f"Contexto: {context}\nPregunta: {question}\nRespuesta:",
                    },
                ],
                max_tokens=settings.LLM_MAX_NEW_TOKENS,
                temperature=0.0,
            )
            output = completion.choices[0].message.content.strip()
            if self._is_invalid_model_output(output, context):
                return (
                    "No pude generar una respuesta útil con el modelo configurado.",
                    False,
                    "Salida del modelo inválida",
                )
            return output, True, None
        except Exception as exc:
            fallback = self._compose_answer(context)
            return fallback, False, str(exc)

    def _persist_history(self, session_id: str, question: str, answer: str) -> None:
        timestamp = datetime.now(timezone.utc).isoformat()
        session = self.collection.find_one({"type": "chat_session", "session_id": session_id})
        messages = session.get("messages", []) if session else []
        messages.append({"role": "user", "content": question, "timestamp": timestamp})
        messages.append({"role": "assistant", "content": answer, "timestamp": timestamp})
        messages = messages[-(settings.CHAT_WINDOW * 2):]
        payload = {
            "type": "chat_session",
            "session_id": session_id,
            "messages": messages,
            "updated_at": timestamp,
        }
        if session:
            self.collection.replace_one({"_id": session["_id"]}, payload, upsert=True)
        else:
            self.collection.insert_one(payload)

    def get_history(self, session_id: str) -> list[dict[str, Any]]:
        session = self.collection.find_one({"type": "chat_session", "session_id": session_id})
        return session.get("messages", []) if session else []

    def get_analytics(self) -> dict[str, Any]:
        total_sessions = self.collection.count_documents({"type": "chat_session"})
        total_messages = 0
        for session in self.collection.find({"type": "chat_session"}):
            total_messages += len(session.get("messages", []))
        return {
            "total_sessions": total_sessions,
            "total_messages": total_messages,
        }
