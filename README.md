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
- MONGO_COLLECTION_NAME
- EMBEDDING_MODEL

## Ejecutar localmente sin Docker

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Limitaciones conocidas

- El scraping es básico y funciona mejor con páginas estáticas o semiestáticas.
- La respuesta del chat es de tipo retrieval-based y no usa un LLM generativo pesado.
- La búsqueda es por similitud de embeddings, no por vector search nativo de MongoDB Atlas.

## Futuras mejoras

- Añadir un reranker para mejorar la calidad del retrieval.
- Integrar un modelo generativo open source de Hugging Face para respuestas más naturales.
- Añadir autenticación y manejo de errores más robusto.
- Guardar métricas de uso y calidad de respuestas en dashboards.
