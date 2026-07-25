FROM python:3.11-slim AS base

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir .

ENV LLM_PROVIDER=demo \
    LOG_LEVEL=INFO

EXPOSE 8000

CMD ["uvicorn", "data_analyst.app.api:app", "--host", "0.0.0.0", "--port", "8000"]
