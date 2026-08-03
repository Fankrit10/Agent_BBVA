from abc import ABC, abstractmethod
from datetime import datetime, timezone


class RagDocumentFactory(ABC):
    @abstractmethod
    def create_document(self, source_url: str, title: str, raw_text: str, cleaned_text: str, chunks: list) -> dict:
        """Create a document payload for persistence."""

    @abstractmethod
    def create_chunk(self, chunk_text: str, chunk_index: int, embedding: list) -> dict:
        """Create a chunk payload with its embedding."""


class StandardRagDocumentFactory(RagDocumentFactory):
    def create_document(self, source_url: str, title: str, raw_text: str, cleaned_text: str, chunks: list) -> dict:
        timestamp = datetime.now(timezone.utc).isoformat()
        return {
            "type": "document",
            "source_url": source_url,
            "title": title,
            "raw_text": raw_text,
            "cleaned_text": cleaned_text,
            "chunks": chunks,
            "created_at": timestamp,
            "updated_at": timestamp,
        }

    def create_chunk(self, chunk_text: str, chunk_index: int, embedding: list) -> dict:
        return {
            "chunk_index": chunk_index,
            "text": chunk_text,
            "embedding": embedding,
        }
