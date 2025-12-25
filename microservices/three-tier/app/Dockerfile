FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

<<<<<<< Updated upstream
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5002", "app:app"]
=======
CMD ["gunicorn", "app:app",
     "-k", "gevent",
     "-w", "4",
     "--worker-connections", "600",
     "--timeout", "60",
     "--bind", "0.0.0.0:5002"]
>>>>>>> Stashed changes
