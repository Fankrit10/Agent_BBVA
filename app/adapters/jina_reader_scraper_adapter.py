import re

import requests

from app.adapters.bs4_scraper_adapter import BeautifulSoupScraperAdapter

READER_ENDPOINT = "https://r.jina.ai/"

MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
MARKDOWN_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")


class JinaReaderScraperAdapter(BeautifulSoupScraperAdapter):
    """Fetches the page through Jina AI's public reader proxy (r.jina.ai),
    which renders the page server-side (its own IPs, its own headless
    browser) and returns clean Markdown. Because the request never reaches
    the target site from this server's network, it routinely gets through
    WAFs (Akamai, etc.) that block automated requests by IP/network
    reputation — the kind of block a local headless browser can't evade
    since the block happens before any page JavaScript runs."""

    def fetch(self, url: str) -> str:
        response = requests.get(f"{READER_ENDPOINT}{url}", timeout=30)
        response.raise_for_status()
        return response.text

    def extract(self, html: str, url: str) -> tuple[str, str]:
        markdown = MARKDOWN_IMAGE_RE.sub("", html)
        markdown = MARKDOWN_LINK_RE.sub(r"\1", markdown)

        raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in markdown.splitlines()]
        raw_lines = [line for line in raw_lines if line]
        raw_text = "\n".join(raw_lines)[:20000]

        cleaned_lines = [line for line in raw_lines if not self._is_noise_line(line)]
        cleaned_text = "\n".join(cleaned_lines)[:20000]
        return raw_text, cleaned_text
