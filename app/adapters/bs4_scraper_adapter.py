import collections
import collections.abc
import re

from bs4 import BeautifulSoup

from app.adapters.base import BaseScraperAdapter


if not hasattr(collections, "Callable"):
    collections.Callable = collections.abc.Callable


class BeautifulSoupScraperAdapter(BaseScraperAdapter):
    def extract(self, html: str, url: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        text = soup.get_text("\n", strip=True)
        lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
        cleaned = "\n".join(line for line in lines if line)
        return cleaned[:20000]
