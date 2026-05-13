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
RUN pip install . --target /packages

# Runtime Stage
FROM gcr.io/distroless/python3-debian13:nonroot
USER 1001:1001
WORKDIR /app

COPY --from=builder /packages /packages

COPY src/ /app/

ENV PYTHONPATH=/packages
ENV NUMBA_CACHE_DIR=/tmp
EXPOSE 8000
CMD ["standalone.py"]