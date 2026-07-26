"""Power BI REST API client - DAX query execution only. No write/admin
operations (e.g. triggering a refresh) are exposed here; the datasource
agent is intentionally read-only.

Every call takes the caller's own delegated access token and sends it as-is
- this client holds no auth state itself (see `clients/powerbi/auth.py`'s
module docstring for why a delegated user token is required: `executeQueries`
rejects a service-principal token outright on any dataset with row-level
security). Queries are always structured `SUMMARIZECOLUMNS(...)`/`ROW(...)` calls built
and validated from a `DaxQuerySpec` (see `dax.py`) - never free-form DAX text
handed in directly.
"""
from __future__ import annotations

import httpx
import pandas as pd

from data_analyst.clients.powerbi.dax import (
    DaxQuerySpec,
    build_dax_query,
    parse_execute_queries_response,
    validate_dax_query,
)
from data_analyst.config.settings import PowerBiCatalog, get_catalog
from data_analyst.telemetry.logging import get_logger
from data_analyst.telemetry.tracing import trace_span

_BASE_URL = "https://api.powerbi.com/v1.0/myorg"
_logger = get_logger("clients.powerbi.rest")


class PBIRestClient:
    def __init__(
        self,
        catalog: PowerBiCatalog | None = None,
        base_url: str = _BASE_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._catalog = catalog or get_catalog()
        self._base_url = base_url
        self._transport = transport
        """`transport` is only ever set in tests, via `httpx.MockTransport` -
        production callers leave it as None."""

    def _client(self, access_token: str) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {access_token}"},
            transport=self._transport,
            timeout=30.0,
        )

    async def run_dax_query(self, access_token: str, spec: DaxQuerySpec) -> tuple[str, pd.DataFrame]:
        """Build, validate, and execute a structured DAX query against the
        Power BI dataset resolved from `spec.model_name` via the catalog
        config.

        Returns the resolved DAX text (for transparency in the tool result)
        and the resulting DataFrame. Raises ValueError if the model is
        unknown, the built query fails structural validation, or Power BI's
        `executeQueries` call itself errors (e.g. an unknown column/table -
        this client can't validate those without a live schema lookup, so
        Power BI's own error is what the agent sees and can react to).
        """
        with trace_span("pbi_rest.run_dax_query", model_name=spec.model_name, table=spec.table):
            model = self._catalog.find_model(spec.model_name)
            if model is None:
                raise ValueError(f"Unknown semantic model '{spec.model_name}'")

            dax_query = build_dax_query(spec)
            validate_dax_query(dax_query, spec)

            body = {"queries": [{"query": dax_query}], "serializerSettings": {"includeNulls": True}}
            async with self._client(access_token) as client:
                response = await client.post(f"/datasets/{model.dataset_id}/executeQueries", json=body)
                if response.status_code >= 400:
                    # INFO, not DEBUG (trace_span's own logging is DEBUG-only,
                    # and LOG_LEVEL defaults to INFO) - httpx's own request
                    # logging shows the status code but never the body, which
                    # is where Power BI's actual reason lives.
                    _logger.info("executeQueries %s failed: query=%s body=%s", response.status_code, dax_query, response.text)
                    raise ValueError(f"Power BI query failed ({response.status_code}): {response.text}")
                df = parse_execute_queries_response(response.json(), spec)
            return dax_query, df
