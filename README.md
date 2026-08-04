# RAG BBVA

Sistema RAG minimalista y funcional en Python con FastAPI, MongoDB, scraping web y embeddings open source de Hugging Face. La app permite:

- Ingresar una URL desde una UI web o elegir una de tres URLs por defecto (bbva.com.co).
- Scrapear el contenido de forma automática incluso en sitios protegidos por WAF anti-bot, limpiarlo y guardar una copia local de lo crudo y lo limpio en `data/raw/` y `data/clean/`.
- Si por alguna razón ninguna estrategia automática consigue el contenido, indexar como último recurso una copia HTML guardada manualmente desde un navegador real, subida por la UI.
- Vectorizar los fragmentos con un modelo open source de Hugging Face e indexarlos en MongoDB (colección `RAG_BBVA`).
- Consultar el contenido indexado desde una segunda pestaña de chat.
- Persistir el historial de conversación por sesión.

## Supuesto de diseño

Se asumió que la base MongoDB está disponible (localmente o vía Docker) y que algunos sitios objetivo (bancos) pueden requerir renderizado JavaScript y/o bloquear tráfico automatizado con un WAF. El scraping por eso encadena varias estrategias en vez de asumir un simple `requests.get`.

## Stack tecnológico

- Python 3.11
- FastAPI para la API y UI minimalista
- MongoDB para persistir documentos y mensajes de chat
- Cadena de estrategias de scraping (ver "Patrones de diseño"): Jina AI Reader (`r.jina.ai`, proxy de lectura gratuito que renderiza el sitio desde su propia infraestructura) → Playwright (Chromium headless, renderiza JavaScript localmente) → BeautifulSoup + `requests` (HTTP simple) — BeautifulSoup limpia y estructura el texto en todos los casos
- sentence-transformers + Hugging Face para embeddings open source
- Hugging Face Inference API (`huggingface_hub.InferenceClient`) con `HuggingFaceH4/zephyr-7b-beta` para la generación de respuestas del chat — modelo open source alojado en Hugging Face, sin dependencias de OpenAI ni de cargar un LLM localmente
- Docker y Docker Compose para levantar la app y la base de datos

## Patrones de diseño implementados

- Singleton: conexión a MongoDB centralizada en app/db/mongo_singleton.py
- Adapter (+ Chain of Responsibility): `BaseScraperAdapter` desacopla "cómo se obtiene y limpia una página" del resto del sistema. Tres adaptadores intercambiables — `JinaReaderScraperAdapter`, `PlaywrightScraperAdapter`, `BeautifulSoupScraperAdapter` — y `ResilientScraperAdapter` los encadena en ese orden, probando el siguiente automáticamente si el anterior no devuelve contenido útil (p. ej. un sitio bloqueado por WAF) — app/adapters/
- Abstract Factory: creación de documentos y chunks centralizada en app/factories/rag_document_factory.py

## Requisitos previos

- Docker Desktop o Docker Engine
- Docker Compose
- Puerto 8000 libre para la app
- Puerto 27017 libre para MongoDB

## Levantar con Docker

Asegúrate de tener el archivo `.env` en la raíz del proyecto con las variables necesarias. Docker Compose cargará esas variables para el servicio de la app.

```bash
docker compose up --build
```

La interfaz quedará disponible en:

- http://localhost:8000/

## Cómo usar

1. Abre la UI en http://localhost:8000/.
2. En la pestaña Scraping, elige una URL por defecto (bbva.com.co) o escribe una personalizada.
3. Pulsa Guardar URL en el RAG para indexar el contenido — funciona de forma automática incluso contra sitios protegidos por WAF (ver "Scraping contra sitios protegidos por WAF" abajo).
4. Si excepcionalmente ninguna estrategia automática consigue el contenido, usa la sección "Sitio bloqueado por anti-bot" como último recurso: abre la URL en tu propio navegador, guárdala con Ctrl+S eligiendo "Página web, solo HTML", e indexa ese archivo `.html` con el botón correspondiente.
5. Cambia a la pestaña Chat y escribe preguntas sobre la información recopilada.
6. Usa el mismo ID de sesión para mantener el contexto de conversaciones anteriores.

### Scraping contra sitios protegidos por WAF (ej. BBVA Colombia)

Se utilizó un proxy para saltar esta restriccion

1. **Jina AI Reader** (`https://r.jina.ai/<url>`) — servicio gratuito que renderiza la página desde su propia infraestructura y devuelve Markdown limpio. Como la petición nunca sale de la red del servidor hacia BBVA, evita el bloqueo por IP. Es la estrategia que consigue el contenido real de bbva.com.co.
2. **Playwright** (Chromium headless local) — para sitios que necesitan JavaScript pero no bloquean por red.
3. **BeautifulSoup + requests** — HTTP simple, para páginas estáticas.

Si las tres fallan (por ejemplo si Jina AI también queda bloqueado en el futuro), queda como respaldo manual la carga de HTML descrita arriba — endpoint `POST /api/index-html-file`.

## Variables de entorno

Se leen automáticamente desde el archivo .env si existe. Los valores por defecto son:

- MONGO_URL
- MONGO_DB_NAME
- MONGO_COLLECTION_NAME (default: `RAG_BBVA`)
- EMBEDDING_MODEL (default: `sentence-transformers/all-MiniLM-L6-v2`)
- CHUNK_SIZE (default: `600`)
- CHAT_WINDOW (default: `6`, cantidad de mensajes previos que se recuerdan por sesión)
- LLM_MODEL (default: `HuggingFaceH4/zephyr-7b-beta`, modelo open source de Hugging Face usado para generar respuestas)
- LLM_MAX_NEW_TOKENS (default: `300`)
- HF_TOKEN (**obligatorio** para que el chat responda con el LLM): token gratuito de Hugging Face con permiso "Read". Créalo en https://huggingface.co/settings/tokens y pégalo en `.env` como `HF_TOKEN=hf_...`. Si falta o el modelo no responde, el sistema cae automáticamente a una respuesta extractiva basada en el fragmento más relevante recuperado (sin cortar el flujo del chat).

## Ejecutar localmente sin Docker

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium  # instala el navegador headless usado para el scraping
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Limitaciones conocidas

- El scraping de bbva.com.co depende de la disponibilidad del proxy gratuito de Jina AI Reader (`r.jina.ai`); si ese servicio cambiara sus condiciones de uso o dejara de estar disponible, el sistema cae automáticamente a Playwright y luego a HTTP simple (que sí serán bloqueados por el WAF de BBVA), y como último recurso queda la carga manual de HTML vía la UI.
- La generación de respuestas depende de la disponibilidad de la Inference API de Hugging Face (modelo `HuggingFaceH4/zephyr-7b-beta`); si el modelo tarda en "despertar" (cold start) o el token no está configurado, la app cae a una respuesta extractiva basada en el fragmento más relevante en vez de fallar.
- La búsqueda es por similitud de embeddings calculada en la app, no por vector search nativo de MongoDB Atlas.

## Futuras mejoras

- Permitir elegir entre varios modelos de Hugging Face desde la UI.
- Añadir autenticación y manejo de errores más robusto.
- Guardar métricas de uso y calidad de respuestas en dashboards.
