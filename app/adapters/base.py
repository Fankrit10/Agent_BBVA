from abc import ABC, abstractmethod


class BaseScraperAdapter(ABC):
    @abstractmethod
    def fetch(self, url: str) -> str:
        """Return the rendered HTML for the given URL."""

    @abstractmethod
    def extract(self, html: str, url: str) -> tuple[str, str]:
        """Return a (raw_text, cleaned_text) tuple extracted from the HTML."""
