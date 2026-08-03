# RAG BBVA

Sistema RAG minimalista y funcional en Python con FastAPI, MongoDB, scraping web y embeddings open source de Hugging Face. La app permite:

- Ingresar una URL desde una UI web o elegir una de tres URLs por defecto.
- Scrapear el contenido, limpiarlo y almacenarlo en MongoDB.
- Vectorizar los fragmentos con un modelo open source de Hugging Face.
- Consultar el contenido indexado desde una segunda pestaña de chat.
- Persistir el historial de conversación por sesión.

## Supuesto de diseño

Se asumió que la base MongoDB está disponible (localmente o vía Docker) y que el contenido a indexar puede ser extraído con scraping clásico sin renderizado JavaScript.

## Stack tecnológico

- Python 3.11
- FastAPI para la API y UI minimalista
- MongoDB para persistir documentos y mensajes de chat
- BeautifulSoup para scraping
- sentence-transformers + Hugging Face para embeddings open source
- Hugging Face Inference API (`huggingface_hub.InferenceClient`) con `HuggingFaceH4/zephyr-7b-beta` para la generación de respuestas del chat — modelo open source alojado en Hugging Face, sin dependencias de OpenAI ni de cargar un LLM localmente
- Docker y Docker Compose para levantar la app y la base de datos

## Patrones de diseño implementados

- Singleton: conexión a MongoDB centralizada en app/db/mongo_singleton.py
- Adapter: extracción de texto desacoplada mediante app/adapters/bs4_scraper_adapter.py
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
2. En la pestaña Scraping, elige una URL por defecto o escribe una personalizada.
3. Pulsa Guardar URL en el RAG para indexar el contenido.
4. Cambia a la pestaña Chat y escribe preguntas sobre la información recopilada.
5. Usa el mismo ID de sesión para mantener el contexto de conversaciones anteriores.

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
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Limitaciones conocidas

- El scraping es básico y funciona mejor con páginas estáticas o semiestáticas.
- La generación de respuestas depende de la disponibilidad de la Inference API de Hugging Face (modelo `HuggingFaceH4/zephyr-7b-beta`); si el modelo tarda en "despertar" (cold start) o el token no está configurado, la app cae a una respuesta extractiva basada en el fragmento más relevante en vez de fallar.
- La búsqueda es por similitud de embeddings calculada en la app, no por vector search nativo de MongoDB Atlas.

## Futuras mejoras

- Añadir un reranker para mejorar la calidad del retrieval.
- Permitir elegir entre varios modelos de Hugging Face desde la UI.
- Añadir autenticación y manejo de errores más robusto.
- Guardar métricas de uso y calidad de respuestas en dashboards.
