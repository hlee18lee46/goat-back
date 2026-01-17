# mongo_repo.py
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, DESCENDING

_client: Optional[MongoClient] = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise RuntimeError("Missing MONGO_URI")
        # reasonable defaults; you can tune later
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
    return _client


def get_collection():
    db_name = os.getenv("MONGO_DB", "kpulse")
    coll_name = os.getenv("MONGO_COLLECTION", "events")
    client = get_client()
    return client[db_name][coll_name]


def ping() -> bool:
    coll = get_collection()
    coll.database.command("ping")
    return True


def insert_event(doc: Dict[str, Any]) -> str:
    coll = get_collection()
    doc = dict(doc)
    doc["created_at"] = datetime.now(timezone.utc).isoformat()
    res = coll.insert_one(doc)
    return str(res.inserted_id)


def read_events(limit: int = 50) -> List[Dict[str, Any]]:
    coll = get_collection()
    docs = list(coll.find({}).sort("created_at", DESCENDING).limit(limit))
    for d in docs:
        d["_id"] = str(d["_id"])
    return docs
