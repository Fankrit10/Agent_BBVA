from pymongo import MongoClient

from app.config import settings


class MongoSingleton:
    _instance = None
    _client = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        if MongoSingleton._client is None:
            MongoSingleton._client = MongoClient(settings.MONGO_URL, serverSelectionTimeoutMS=5000)
            MongoSingleton._client.admin.command("ping")
        self.db = MongoSingleton._client[settings.MONGO_DB_NAME]

    def get_collection(self, name=None):
        return self.db[name or settings.MONGO_COLLECTION_NAME]

    def close(self):
        if MongoSingleton._client is not None:
            MongoSingleton._client.close()
            MongoSingleton._client = None
            MongoSingleton._instance = None
