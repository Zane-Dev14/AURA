FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["gunicorn", "app:app", \
     "--worker-class", "gevent", \
     "--workers", "2", \
     "--worker-connections", "1000", \
     "--timeout", "120", \
     "--bind", "0.0.0.0:5002"]