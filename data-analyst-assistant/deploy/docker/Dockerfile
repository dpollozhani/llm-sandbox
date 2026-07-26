FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

ENV LOG_LEVEL=INFO
# LLM_PROVIDER (and the matching ANTHROPIC_API_KEY/AZURE_OPENAI_* vars) is
# required at runtime - no default here, since there's no fallback provider
# to silently fall back to. Pass it via `docker run -e`, docker-compose, or
# the hosting platform's environment settings.

EXPOSE 8000

# Shell form (with `exec`, so uvicorn becomes PID 1 and gets signals directly)
# so $PORT expands - Render (and similar PaaS) assign it at runtime; defaults
# to 8000 for local `docker run`/docker-compose where PORT isn't set.
CMD exec uvicorn data_analyst.app.api:app --host 0.0.0.0 --port ${PORT:-8000}
