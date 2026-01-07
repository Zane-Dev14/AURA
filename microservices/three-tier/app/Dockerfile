FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app",     "-k", "gevent",     "-w", "4",     "--worker-connections", "600",     "--timeout", "60",     "--bind", "0.0.0.0:5002"]
