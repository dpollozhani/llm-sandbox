"""Datasource agent: read-only PBI MCP + PBI REST tools, plus the graph nodes
that use them. No tool here mutates anything in Power BI - metadata lookups
and structured DAX queries only.

Every PBI tool reads the current request's delegated access token from
`InjectedState` (`pbi_token`, set by `app/api.py` from whichever the caller
signed in with - see `clients/powerbi/auth.py`'s module docstring for why a
delegated user token is required at all: row-level security depends on
whose identity the call runs as, so there's no app-only/service-principal
fallback here) rather than holding any auth of their own.
"""
from __future__ import annotations

from typing import Annotated

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState, ToolNode

from data_analyst.agents.common.tools import flag_ambiguity
from data_analyst.agents.datasource.models import DataSourceQueryResult
from data_analyst.agents.datasource.prompts import SYSTEM_PROMPT
from data_analyst.agents.datasource.state import DatasourceState
from data_analyst.clients.powerbi.dax import DaxColumn, DaxFilter, DaxMeasure, DaxQuerySpec
from data_analyst.clients.powerbi.mcp import PBIMcpClient, get_metadata_cache
from data_analyst.clients.powerbi.rest import PBIRestClient
from data_analyst.clients.sandbox.client import get_sandbox_client
from data_analyst.config.settings import Glossary, PowerBiCatalog, inject_glossary
from data_analyst.utils.dataframe import preview_records

_NOT_SIGNED_IN = "Not signed in with Power BI access for this - ask the user to sign in again (/auth/login)."


def _describe(exc: BaseException) -> str:
    """A real message for `exc`, unwrapping `ExceptionGroup`s (the `mcp`
    SDK's transport, and httpx, both run background tasks in an anyio task
    group - a connection/protocol failure there surfaces as a bare
    "unhandled errors in a TaskGroup (N sub-exceptions)" otherwise, with the
    actual cause hidden a level down)."""
    if isinstance(exc, ExceptionGroup):
        return "; ".join(_describe(e) for e in exc.exceptions)
    return str(exc) or repr(exc)


def build_tools(mcp_client: PBIMcpClient | None = None, rest_client: PBIRestClient | None = None) -> list[BaseTool]:
    """Builds the datasource agent's tools bound to `mcp_client`/`rest_client`
    (real clients by default; tests inject fakes here instead of reaching
    past the `@tool` decorators)."""
    mcp = mcp_client or PBIMcpClient()
    rest = rest_client or PBIRestClient()

    @tool
    async def pbi_mcp_get_semantic_metadata(model_name: str, state: Annotated[DatasourceState, InjectedState]) -> dict:
        """Get a semantic model's schema (tables, columns, measures,
        relationships) via the Power BI MCP server's semantic metadata tool.
        Call this for a model before querying it if you haven't already
        seen its schema this conversation."""
        token = state.get("pbi_token")
        if not token:
            return {"error": _NOT_SIGNED_IN}
        cache = get_metadata_cache(state["session_id"])
        if (cached := cache.get(model_name)) is not None:
            # A fresh specialist subgraph is rebuilt on every delegation and
            # has no memory of a schema it already fetched earlier in this
            # conversation - this cache is what actually makes the "unless
            # you've already seen it" guidance in this agent's system prompt
            # true across delegations, not just within one.
            return cached
        try:
            metadata = await mcp.get_semantic_metadata(token, model_name)
        except Exception as exc:  # noqa: BLE001 - network/protocol boundary, see module docstring
            return {"error": f"Power BI MCP call failed: {_describe(exc)}"}
        cache.remember(model_name, metadata)
        return metadata

    @tool
    async def pbi_rest_run_dax_query(
        model_name: str,
        group_by: list[DaxColumn],
        filters: list[DaxFilter],
        measures: list[DaxMeasure],
        state: Annotated[DatasourceState, InjectedState],
    ) -> dict:
        """Run a structured query against a Power BI semantic model via the PBI REST API.

        The query is always built from `group_by` columns, `filters`, and
        `measures` - never free-form DAX text - and is validated before
        being sent. Pass empty lists for anything not needed, but at least
        one of `group_by` or `measures` is required. With `group_by`, it's a
        single SUMMARIZECOLUMNS(...) call; with none (a grand total, not
        broken out by anything), it's a ROW(...) call instead.

        Each `group_by`/`filters` entry, and an ad-hoc `measures` aggregation,
        names its own table - a group-by column and an aggregated measure can
        come from different, related tables in the same query (e.g. group by
        a dimension table's column while summing a fact table's column).
        Don't force everything onto one table.

        If this exact query (same columns/filters/measures) was already
        run earlier in this conversation, the cached result is reused instead
        of issuing a new query - check the `reused` field in the response.

        Returns a preview of the resulting rows, the `group_by`/`filters`/
        `measures` actually used (relay these to the user for transparency
        about what was fetched), and a `dataset_id` that the analysis agent
        can use to load the full result as a DataFrame.
        """
        token = state.get("pbi_token")
        if not token:
            return {"error": _NOT_SIGNED_IN}

        try:
            spec = DaxQuerySpec(model_name=model_name, group_by=group_by, filters=filters, measures=measures)
        except ValueError as exc:
            return {"error": str(exc)}

        # Plain `table.column` form for the user-facing group_by/filters/measures
        # fields - computed once, shared by both return branches below.
        query_group_by = [f"{c.table}.{c.column}" for c in spec.group_by]
        query_filters = [f"{f.table}.{f.column} {f.operator} {f.value!r}" for f in spec.filters]
        query_measures = [
            m.name if m.aggregation is None else f"{m.name} = {m.aggregation}({m.table}.{m.column})"
            for m in spec.measures
        ]

        store = get_sandbox_client(state["session_id"])
        cache_key = spec.cache_key()
        cached_dataset_id = store.find_cached(cache_key)
        if cached_dataset_id is not None:
            df = store.peek(cached_dataset_id)
            return DataSourceQueryResult(
                dataset_id=cached_dataset_id,
                model_name=model_name,
                group_by=query_group_by,
                filters=query_filters,
                measures=query_measures,
                row_count=len(df),
                preview=preview_records(df),
                reused=True,
            ).model_dump()

        try:
            dax_query, df = await rest.run_dax_query(token, spec)
        except Exception as exc:  # noqa: BLE001 - network/protocol boundary, see module docstring
            return {"error": f"Power BI query failed: {_describe(exc)}"}

        dataset_id = store.stage(df)
        store.remember(cache_key, dataset_id)
        return DataSourceQueryResult(
            dataset_id=dataset_id,
            model_name=model_name,
            group_by=query_group_by,
            filters=query_filters,
            measures=query_measures,
            row_count=len(df),
            preview=preview_records(df),
            reused=False,
            dax_query=dax_query,
        ).model_dump()

    return [
        pbi_mcp_get_semantic_metadata,
        pbi_rest_run_dax_query,
        flag_ambiguity,
    ]


def build_agent_node(
    llm: BaseChatModel,
    tools: list[BaseTool] | None = None,
    catalog: PowerBiCatalog | None = None,
    glossary: Glossary | None = None,
):
    tools = tools if tools is not None else build_tools()
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = SYSTEM_PROMPT
    if catalog is not None and catalog.semantic_models:
        # There's no "list models" tool (removed along with workspace
        # listing/refresh history - out of scope for this build), so this is
        # the only way the model learns which `model_name` values are valid
        # to pass to pbi_mcp_get_semantic_metadata/pbi_rest_run_dax_query.
        names = ", ".join(f'"{m.model_name}"' for m in catalog.semantic_models)
        system_prompt += f"\n\nAvailable semantic models: {names}."
    system_prompt = inject_glossary(system_prompt, glossary)

    async def agent_node(state: DatasourceState):
        response = await llm_with_tools.ainvoke([SystemMessage(content=system_prompt), *state["messages"]])
        return {"messages": [response]}

    return agent_node


def build_tool_node(tools: list[BaseTool] | None = None) -> ToolNode:
    return ToolNode(tools if tools is not None else build_tools())
