import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from app.adapters.playwright_scraper_adapter import PlaywrightScraperAdapter
from app.config import settings
from app.db.mongo_singleton import MongoSingleton
from app.factories.rag_document_factory import StandardRagDocumentFactory

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class ScraperService:
    def __init__(self):
        self.adapter = PlaywrightScraperAdapter()
        self.factory = StandardRagDocumentFactory()
        self.embedding_model = None
        self.collection = MongoSingleton.get_instance().get_collection()

    def _load_model(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self.embedding_model

    def fetch_page(self, url: str) -> dict[str, Any]:
        html = self.adapter.fetch(url)
        return {
            "url": url,
            "html": html,
            "title": self._extract_title(html),
        }

    def _extract_title(self, html: str) -> str:
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if not match:
            return "Sin título"
        return re.sub(r"\s+", " ", match.group(1)).strip()

    def _persist_local_copies(self, source_hash: str, raw_html: str, raw_text: str, cleaned_text: str) -> None:
        raw_dir = DATA_DIR / "raw"
        clean_dir = DATA_DIR / "clean"
        raw_dir.mkdir(parents=True, exist_ok=True)
        clean_dir.mkdir(parents=True, exist_ok=True)
        (raw_dir / f"{source_hash}.html").write_text(raw_html, encoding="utf-8")
        (raw_dir / f"{source_hash}.txt").write_text(raw_text, encoding="utf-8")
        (clean_dir / f"{source_hash}.txt").write_text(cleaned_text, encoding="utf-8")

    def scrape_and_index(self, url: str) -> dict[str, Any]:
        page = self.fetch_page(url)
        raw_text, cleaned_text = self.adapter.extract(page["html"], page["url"])
        source_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()
        self._persist_local_copies(source_hash, page["html"], raw_text, cleaned_text)

        chunks = self._chunk_text(cleaned_text)
        chunk_payloads = []
        for index, chunk in enumerate(chunks):
            embedding = self._embed_text(chunk)
            chunk_payloads.append(self.factory.create_chunk(chunk, index, embedding))

        document_payload = self.factory.create_document(
            source_url=url,
            title=page["title"],
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            chunks=chunk_payloads,
        )
        document_payload["source_hash"] = source_hash

        existing = self.collection.find_one({"source_hash": document_payload["source_hash"]})
        if existing:
            self.collection.replace_one({"_id": existing["_id"]}, document_payload, upsert=True)
            action = "updated"
        else:
            self.collection.insert_one(document_payload)
            action = "inserted"

        return {
            "status": "ok",
            "action": action,
            "source_url": url,
            "title": page["title"],
            "chunks_count": len(chunk_payloads),
        }

    def _is_noise_chunk(self, chunk: str) -> bool:
        alpha_chars = sum(1 for ch in chunk if ch.isalpha())
        if alpha_chars / max(1, len(chunk)) < 0.45:
            return True
        if len(re.findall(r"[\{\}\[\]<>]", chunk)) > len(chunk) * 0.2:
            return True
        return False

    def _chunk_text(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
        chunks = []
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) < settings.CHUNK_SIZE:
                current = f"{current} {paragraph}".strip()
            else:
                if current and not self._is_noise_chunk(current):
                    chunks.append(current)
                current = paragraph
        if current and not self._is_noise_chunk(current):
            chunks.append(current)
        return [chunk for chunk in chunks if len(chunk) > 30 and not self._is_noise_chunk(chunk)]

    def _embed_text(self, text: str) -> list[float]:
        model = self._load_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.astype(float).tolist()

    def retrieve_relevant_chunks(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        query_embedding = self._embed_text(question)
        documents = list(self.collection.find({"type": "document"}))
        scored_chunks = []
        for document in documents:
            for chunk in document.get("chunks", []):
                embedding = chunk.get("embedding")
                if not embedding:
                    continue
                score = self._cosine_similarity(query_embedding, embedding)
                scored_chunks.append(
                    {
                        "score": float(score),
                        "text": chunk.get("text", ""),
                        "source_url": document.get("source_url"),
                    }
                )

        scored_chunks.sort(key=lambda item: item["score"], reverse=True)
        return scored_chunks[:top_k]

    def _cosine_similarity(self, a: list[float], b: list[float]) -> float:
        vector_a = np.array(a, dtype=float)
        vector_b = np.array(b, dtype=float)
        if np.linalg.norm(vector_a) == 0 or np.linalg.norm(vector_b) == 0:
            return 0.0
        return float(np.dot(vector_a, vector_b) / (np.linalg.norm(vector_a) * np.linalg.norm(vector_b)))
