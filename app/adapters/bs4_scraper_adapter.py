import re

import requests
from bs4 import BeautifulSoup

from app.adapters.base import BaseScraperAdapter


class BeautifulSoupScraperAdapter(BaseScraperAdapter):
    """Fetches a page with a plain HTTP request. Works only for
    server-rendered (non-JavaScript) pages."""

    def fetch(self, url: str) -> str:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        return response.text

    def _is_noise_line(self, line: str) -> bool:
        if len(line) < 30:
            return True
        if re.match(r'^[\W_]+$', line):
            return True
        alpha_chars = sum(1 for ch in line if ch.isalpha())
        if alpha_chars / max(1, len(line)) < 0.45:
            return True
        if re.search(r'[\{\}\[\]\<\>]{3,}', line):
            return True
        if re.search(r'"\s*[,\]}]', line) or re.search(r"'\s*[,\]}]", line):
            return True
        return False

    def extract(self, html: str, url: str) -> tuple[str, str]:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "header", "footer", "nav", "form", "input", "button", "meta", "link"]):
            tag.decompose()

        # separator="\n" preserves one line per text node so paragraphs stay
        # distinguishable; joining with a plain space would collapse the
        # whole page into a single line and drop most of the real content
        # once line-level noise filtering runs.
        text = soup.get_text("\n", strip=True)
        raw_lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
        raw_lines = [line for line in raw_lines if line]
        raw_text = "\n".join(raw_lines)[:20000]

        cleaned_lines = [line for line in raw_lines if not self._is_noise_line(line)]
        cleaned_text = "\n".join(cleaned_lines)[:20000]
        return raw_text, cleaned_text
