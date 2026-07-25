FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

ENV LLM_PROVIDER=demo \
    LOG_LEVEL=INFO

EXPOSE 8000

# Shell form (with `exec`, so uvicorn becomes PID 1 and gets signals directly)
# so $PORT expands - Render (and similar PaaS) assign it at runtime; defaults
# to 8000 for local `docker run`/docker-compose where PORT isn't set.
CMD exec uvicorn data_analyst.app.api:app --host 0.0.0.0 --port ${PORT:-8000}
