from flask import Flask, jsonify, request
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import os

api = Flask(__name__)

# Database configuration (matches your target version)
DB_HOST = os.getenv("DB_HOST", "db")
DB_USER = os.getenv("DB_USER", "user")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_NAME = os.getenv("DB_NAME", "quotesdb")

# SQLAlchemy engine with connection pooling
engine = create_engine(
    f"mysql+mysqldb://{DB_USER}:{DB_PASSWORD}@{DB_HOST}/{DB_NAME}",
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=30,
    pool_recycle=180,
    pool_pre_ping=True,
)

# --------------------
# Routes
# --------------------

@api.route("/api/quotes", methods=["GET"])
def get_quotes():
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT * FROM quotes")).fetchall()
        return jsonify([
            {"id": r[0], "quote": r[1], "author": r[2]}
            for r in rows
        ])

@api.route("/api/quotes", methods=["POST"])
def add_quote():
    data = request.get_json()
    quote = data["quote"]
    author = data["author"]

    with engine.begin() as conn:
        result = conn.execute(
            text("INSERT INTO quotes (quote, author) VALUES (:quote, :author)"),
            {"quote": quote, "author": author}
        )
        quote_id = result.lastrowid

    return jsonify({
        "id": quote_id,
        "quote": quote,
        "author": author
    }), 201

@api.route("/health", methods=["GET"])
def health():
    return "OK", 200

# --------------------
# App entry point
# --------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    api.run(host="0.0.0.0", port=port)
