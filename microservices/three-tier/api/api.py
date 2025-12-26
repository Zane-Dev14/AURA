from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import os
import time
from threading import Lock

api = Flask(__name__)

# Database configuration
DB_HOST = os.getenv("DB_HOST", "db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "quotesdb")

# SQLAlchemy engine with optimized pooling
engine = create_engine(
    f"mysql+mysqldb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    poolclass=QueuePool,
    pool_size=100,           # Increased
    max_overflow=100,        # Increased
    pool_timeout=10,
    pool_recycle=1800,
    pool_pre_ping=True,
    connect_args={"connect_timeout": 5},
)

# Simple in-memory cache (industry standard pattern)
_cache = {"quotes": None, "timestamp": 0}
_cache_lock = Lock()
CACHE_TTL = 10  # 10 seconds

def get_cached_quotes():
    """Get quotes from cache or DB"""
    now = time.time()

    with _cache_lock:
        # Check if cache is valid
        if _cache["quotes"] and (now - _cache["timestamp"]) < CACHE_TTL:
            return _cache["quotes"]

    # Cache miss - fetch from DB
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT id, quote, author FROM quotes LIMIT 100")).fetchall()
        result = [{"id": r[0], "quote": r[1], "author": r[2]} for r in rows]

    # Update cache
    with _cache_lock:
        _cache["quotes"] = result
        _cache["timestamp"] = now

    return result

def invalidate_cache():
    """Clear cache when data changes"""
    with _cache_lock:
        _cache["quotes"] = None
        _cache["timestamp"] = 0

@api.route("/api/quotes", methods=["GET"])
def get_quotes():
    return jsonify(get_cached_quotes())

@api.route("/api/quotes", methods=["POST"])
def add_quote():
    data = request.get_json()
    quote = data.get("quote", "")
    author = data.get("author", "")

    with engine.begin() as conn:
        result = conn.execute(
            text("INSERT INTO quotes (quote, author) VALUES (:quote, :author)"),
            {"quote": quote, "author": author}
        )
        quote_id = result.lastrowid

    # Invalidate cache after write
    invalidate_cache()

    return jsonify({"id": quote_id, "quote": quote, "author": author}), 201

@api.route("/health", methods=["GET"])
def health():
    return "OK", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    api.run(host="0.0.0.0", port=port)
