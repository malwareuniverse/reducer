# Build Stage
FROM docker.io/library/python:3.13-slim-bookworm as builder
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Build-Tools nur im Builder installieren
RUN apt-get update && apt-get install -y \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY pyproject.toml /app/
COPY src/ /app/

RUN pip install --upgrade pip setuptools wheel
RUN pip install .

# Runtime Stage
FROM docker.io/library/python:3.13-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
RUN pip install --upgrade pip setuptools

# Nur die installierten Pakete kopieren
COPY --from=builder /usr/local /usr/local 

COPY src/ /app/
WORKDIR /app

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]