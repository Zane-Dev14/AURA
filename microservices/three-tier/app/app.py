print("### NEW APP VERSION LOADED ###")

from flask import Flask, render_template, request, redirect, url_for
import requests
import os

app = Flask(__name__)

API_URL = os.environ.get("API_URL", "http://api:8080")

SESSION = requests.Session()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            SESSION.post(
                f"{API_URL}/api/quotes",
                json={
                    "quote": request.form.get("quote", ""),
                    "author": request.form.get("author", ""),
                },
                timeout=0.5,
            )
        except Exception:
            # swallow errors — UI must not block
            pass

        return redirect(url_for("index"))

    # GET
    quotes = []
    try:
        resp = SESSION.get(
            f"{API_URL}/api/quotes",
            timeout=0.5,
        )
        if resp.ok:
            quotes = resp.json()
    except Exception:
        # graceful degradation
        quotes = []

    return render_template("index.html", quotes=quotes)


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200
