FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mushroom mushroom
COPY scripts scripts
COPY app app
COPY data data

# Train during the build, so the repository holds no binary model.
RUN python -m scripts.train

# Hugging Face Spaces uses 7860; Render and others supply $PORT.
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port $PORT"]
