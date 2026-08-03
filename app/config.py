import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017/")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "admin")
    MONGO_COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "RAG_BBVA")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    DEFAULT_URLS = [
        "https://www.bbva.com.co/",
        "https://www.bbva.com.co/personas/",
        "https://www.bbva.com.co/empresas/",
    ]
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
    CHAT_WINDOW = int(os.getenv("CHAT_WINDOW", "6"))
    LLM_MODEL = os.getenv("LLM_MODEL", "HuggingFaceH4/zephyr-7b-beta").strip(" '\"")
    HF_TOKEN = os.getenv("HF_TOKEN", "").strip(" '\"")
    LLM_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "300"))


settings = Settings()
