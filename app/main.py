import re

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from pathlib import Path

from app.config import settings
from app.services.chat_service import ChatService
from app.services.scraper_service import ScraperService

CHARSET_RE = re.compile(rb'charset=["\']?\s*([\w-]+)', re.IGNORECASE)


def _decode_html(raw_bytes: bytes) -> str:
    match = CHARSET_RE.search(raw_bytes[:2048])
    declared_encoding = match.group(1).decode("ascii", errors="ignore") if match else None
    for encoding in filter(None, [declared_encoding, "utf-8"]):
        try:
            return raw_bytes.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw_bytes.decode("latin-1")

app = FastAPI(title="BBVA RAG Agent", version="1.0.0")
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.on_event("startup")
def startup_event():
    scraper_service = ScraperService()
    chat_service = ChatService(scraper_service)
    app.state.scraper_service = scraper_service
    app.state.chat_service = chat_service


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/default-urls")
def default_urls() -> dict[str, list[str]]:
    return {"urls": settings.DEFAULT_URLS}


@app.post("/api/scrape-and-index")
async def scrape_and_index(request: Request) -> dict[str, object]:
    payload = await request.json()
    url = payload.get("url", "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="Se requiere una URL")

    scraper_service = getattr(app.state, "scraper_service", None)
    if scraper_service is None:
        raise HTTPException(status_code=500, detail="Servicio no inicializado")

    try:
        result = await run_in_threadpool(scraper_service.scrape_and_index, url)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@app.post("/api/index-html-file")
async def index_html_file(source_url: str = Form(...), file: UploadFile = File(...)) -> dict[str, object]:
    source_url = source_url.strip()
    if not source_url:
        raise HTTPException(status_code=400, detail="Se requiere la URL de origen")

    scraper_service = getattr(app.state, "scraper_service", None)
    if scraper_service is None:
        raise HTTPException(status_code=500, detail="Servicio no inicializado")

    raw_bytes = await file.read()
    html = _decode_html(raw_bytes)

    try:
        result = await run_in_threadpool(scraper_service.index_uploaded_html, source_url, html)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result


@app.post("/api/chat")
async def chat(request: Request) -> dict[str, object]:
    payload = await request.json()
    session_id = payload.get("session_id", "default-session")
    question = payload.get("question", "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Se requiere una pregunta")

    chat_service = getattr(app.state, "chat_service", None)
    if chat_service is None:
        raise HTTPException(status_code=500, detail="Servicio de chat no inicializado")

    return await run_in_threadpool(chat_service.answer_question, session_id, question)


@app.get("/api/history/{session_id}")
def history(session_id: str) -> dict[str, object]:
    chat_service = getattr(app.state, "chat_service", None)
    if chat_service is None:
        raise HTTPException(status_code=500, detail="Servicio de chat no inicializado")
    return {"session_id": session_id, "messages": chat_service.get_history(session_id)}


@app.get("/api/analytics")
def analytics() -> dict[str, object]:
    chat_service = getattr(app.state, "chat_service", None)
    if chat_service is None:
        raise HTTPException(status_code=500, detail="Servicio de chat no inicializado")
    return chat_service.get_analytics()
