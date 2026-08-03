import collections
import collections.abc
import re

from bs4 import BeautifulSoup

from app.adapters.base import BaseScraperAdapter


if not hasattr(collections, "Callable"):
    collections.Callable = collections.abc.Callable


class BeautifulSoupScraperAdapter(BaseScraperAdapter):
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

    def extract(self, html: str, url: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe", "header", "footer", "nav", "form", "input", "button", "meta", "link"]):
            tag.decompose()

        text = soup.get_text(" ", strip=True)
        lines = []
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if not line:
                continue
            if self._is_noise_line(line):
                continue
            lines.append(line)

        cleaned = "\n".join(lines)
        return cleaned[:20000]
