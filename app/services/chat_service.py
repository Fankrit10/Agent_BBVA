from datetime import datetime, timezone
from typing import Any

from app.config import settings
from app.db.mongo_singleton import MongoSingleton


class ChatService:
    def __init__(self, scraper_service):
        self.scraper_service = scraper_service
        self.collection = MongoSingleton.get_instance().get_collection()

    def answer_question(self, session_id: str, question: str) -> dict[str, Any]:
        relevant_chunks = self.scraper_service.retrieve_relevant_chunks(question, top_k=3)
        if not relevant_chunks:
            answer = "Todavía no hay contenido indexado en el RAG. Primero guarda una URL para cargar el conocimiento."
            sources = []
        else:
            best_chunk = relevant_chunks[0]
            answer = self._compose_answer(question, best_chunk["text"])
            sources = [
                {"url": item["source_url"], "score": round(item["score"], 3)} for item in relevant_chunks
            ]

        self._persist_history(session_id, question, answer)
        return {
            "answer": answer,
            "sources": sources,
            "session_id": session_id,
        }

    def _compose_answer(self, question: str, context: str) -> str:
        cleaned_context = context.strip().replace("\n", " ")
        if len(cleaned_context) > 650:
            cleaned_context = cleaned_context[:650] + "..."
        return f"Según la información indexada, {cleaned_context}"

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
