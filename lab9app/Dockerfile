# lab9app/Dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY lab9app.py .
COPY reddit_model_pipeline.joblib .

EXPOSE 5050

CMD ["uvicorn", "lab9app:app", "--host", "0.0.0.0", "--port", "5050"]
