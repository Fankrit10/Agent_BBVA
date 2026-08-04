from playwright.sync_api import sync_playwright

from app.adapters.bs4_scraper_adapter import BeautifulSoupScraperAdapter


class PlaywrightScraperAdapter(BeautifulSoupScraperAdapter):
    """Renders the page in a headless browser before extracting text, so
    content injected by client-side JavaScript (common in bank sites built
    with React/Angular) is captured. Raises on failure (missing browser
    binaries, navigation error, network block) so callers such as
    ResilientScraperAdapter can fall back to another strategy."""

    def fetch(self, url: str) -> str:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            try:
                page = browser.new_page(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    )
                )
                page.goto(url, timeout=30000, wait_until="networkidle")
                return page.content()
            finally:
                browser.close()
