from app.adapters.base import BaseScraperAdapter
from app.adapters.bs4_scraper_adapter import BeautifulSoupScraperAdapter
from app.adapters.jina_reader_scraper_adapter import JinaReaderScraperAdapter
from app.adapters.playwright_scraper_adapter import PlaywrightScraperAdapter

MIN_USEFUL_CHARS = 200


class ResilientScraperAdapter(BaseScraperAdapter):
    """Tries a sequence of scraping strategies (Adapter pattern) until one
    produces real content, so a single blocked network path doesn't fail
    the whole request:

    1. JinaReaderScraperAdapter — fetched from Jina AI's infrastructure,
       bypasses WAFs that block this server's own IP (bank sites).
    2. PlaywrightScraperAdapter — local headless browser, renders
       JavaScript for sites that need it but aren't network-blocked.
    3. BeautifulSoupScraperAdapter — plain HTTP request, fastest for
       simple static/server-rendered pages.
    """

    def __init__(self):
        self._strategies: list[BaseScraperAdapter] = [
            JinaReaderScraperAdapter(),
            PlaywrightScraperAdapter(),
            BeautifulSoupScraperAdapter(),
        ]
        self._last_used = self._strategies[0]

    def fetch(self, url: str) -> str:
        errors = []
        for strategy in self._strategies:
            try:
                content = strategy.fetch(url)
                _, cleaned = strategy.extract(content, url)
                if len(cleaned) >= MIN_USEFUL_CHARS:
                    self._last_used = strategy
                    return content
                errors.append(f"{type(strategy).__name__}: contenido insuficiente ({len(cleaned)} caracteres)")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{type(strategy).__name__}: {exc}")
        raise RuntimeError(
            "No se pudo obtener contenido útil de la URL con ninguna estrategia de scraping. "
            + " | ".join(errors)
        )

    def extract(self, html: str, url: str) -> tuple[str, str]:
        return self._last_used.extract(html, url)
