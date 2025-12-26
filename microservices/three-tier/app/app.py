print("### NEW APP VERSION LOADED ###")
from gevent import monkey
monkey.patch_all()

from flask import Flask, render_template, request, redirect, url_for
import requests
import os

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

app = Flask(__name__)

API_URL = os.environ.get("API_URL", "http://api:8080")

# ---- Safe retry configuration ----
retry_strategy = Retry(
    total=1,
    status_forcelist=[502, 503],
    backoff_factor=0.05,
    raise_on_status=False,
)

adapter = HTTPAdapter(max_retries=retry_strategy)

SESSION = requests.Session()
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)
# ----------------------------------


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
                timeout=2.0,
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
            timeout=2.0,
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
