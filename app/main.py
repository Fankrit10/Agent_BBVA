from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from app.config import settings
from app.services.chat_service import ChatService
from app.services.scraper_service import ScraperService

app = FastAPI(title="BBVA RAG Agent", version="1.0.0")
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


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
        result = scraper_service.scrape_and_index(url)
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

    return chat_service.answer_question(session_id, question)


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
