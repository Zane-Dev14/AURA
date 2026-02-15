from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import os
import time
from threading import Lock

api = FastAPI()

# DB config
DB_HOST = os.getenv("DB_HOST", "db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "quotesdb")

engine = create_engine(
    f"mysql+mysqldb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_timeout=10,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
)

_cache = {"quotes": None, "timestamp": 0}
_cache_lock = Lock()
CACHE_TTL = 10


class QuoteIn(BaseModel):
    text: str
    author: str | None = None


def get_cached_quotes():
    now = time.time()
    with _cache_lock:
        if _cache["quotes"] and now - _cache["timestamp"] < CACHE_TTL:
            return _cache["quotes"]

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text("SELECT id, text, author FROM quotes LIMIT 100")
            ).fetchall()

        result = [{"id": r.id, "text": r.text, "author": r.author} for r in rows]

        with _cache_lock:
            _cache["quotes"] = result
            _cache["timestamp"] = now

        return result
    except SQLAlchemyError:
        return []


def invalidate_cache():
    with _cache_lock:
        _cache["quotes"] = None
        _cache["timestamp"] = 0


@api.get("/api/quotes")
def get_quotes():
    return get_cached_quotes()


@api.post("/api/quotes", status_code=201)
def add_quote(q: QuoteIn):
    if not q.text.strip():
        raise HTTPException(status_code=400, detail="Quote text is required")

    try:
        with engine.begin() as conn:
            result = conn.execute(
                text("INSERT INTO quotes (text, author) VALUES (:text, :author)"),
                {"text": q.text, "author": q.author},
            )

        invalidate_cache()
        return {"id": result.lastrowid, "text": q.text, "author": q.author}

    except SQLAlchemyError:
        raise HTTPException(status_code=500, detail="Database error")


@api.get("/health")
def health():
    return {"status": "ok"}
