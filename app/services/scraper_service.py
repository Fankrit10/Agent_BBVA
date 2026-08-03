import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

from app.adapters.bs4_scraper_adapter import BeautifulSoupScraperAdapter
from app.config import settings
from app.db.mongo_singleton import MongoSingleton
from app.factories.rag_document_factory import StandardRagDocumentFactory


class ScraperService:
    def __init__(self):
        self.adapter = BeautifulSoupScraperAdapter()
        self.factory = StandardRagDocumentFactory()
        self.embedding_model = None
        self.collection = MongoSingleton.get_instance().get_collection()

    def _load_model(self):
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
        return self.embedding_model

    def fetch_page(self, url: str) -> dict[str, Any]:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return {
            "url": url,
            "html": response.text,
            "title": self._extract_title(response.text),
        }

    def _extract_title(self, html: str) -> str:
        match = re.search(r"<title>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if not match:
            return "Sin título"
        return re.sub(r"\s+", " ", match.group(1)).strip()

    def scrape_and_index(self, url: str) -> dict[str, Any]:
        page = self.fetch_page(url)
        cleaned_text = self.adapter.extract(page["html"], page["url"])
        raw_text = cleaned_text[:15000]
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
        document_payload["source_hash"] = hashlib.sha256(url.encode("utf-8")).hexdigest()

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

    def _chunk_text(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n+", text) if part.strip()]
        chunks = []
        current = ""
        for paragraph in paragraphs:
            if len(current) + len(paragraph) < settings.CHUNK_SIZE:
                current = f"{current} {paragraph}".strip()
            else:
                if current:
                    chunks.append(current)
                current = paragraph
        if current:
            chunks.append(current)
        return [chunk for chunk in chunks if len(chunk) > 30]

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
