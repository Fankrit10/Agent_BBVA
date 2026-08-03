from abc import ABC, abstractmethod


class BaseScraperAdapter(ABC):
    @abstractmethod
    def extract(self, html: str, url: str) -> str:
        """Return cleaned text from the raw HTML."""
