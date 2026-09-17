FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app/backend

COPY backend/requirements.txt backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend/ backend/
COPY frontend/ frontend/
COPY Makefile .

# Generate data and train at build time so the image is demo-ready on start.
RUN python -m careweave.data.generator.run --members 2000 \
 && python -m careweave.ml.train \
 && python -m careweave.rag.evaluate \
 && python -m careweave.nlp.evaluate

EXPOSE 8000
CMD ["python", "-m", "uvicorn", "careweave.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "backend"]
