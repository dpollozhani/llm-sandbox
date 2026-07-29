# Working conventions for this project

## Prefer the framework's own opinionated shapes over custom ones

Minimize custom structures. Before adding a new class, function, or
abstraction layer, check whether Pydantic, LangChain, or LangGraph already
gives you the shape you need:

- **Data shapes**: a plain `pydantic.BaseModel` beats a hand-built `dict`
  literal or a bespoke wrapper class - validation, `.model_dump()`, and
  schema generation (for `@tool` args) all come for free. Don't define two
  near-identical models for what's really one concept (e.g. a tool's own
  result shape and a "summary of that result" type elsewhere) - merge them
  if the fields genuinely overlap, and only split when the two really are
  used differently by different callers.
- **State**: LangGraph's `TypedDict` state + `Annotated[..., reducer]`
  pattern (see `agents/common/state.py::ChatState`) is the idiomatic way to
  model graph state - don't reinvent a state container. Plain dicts are
  still correct for fields that must survive checkpointing across
  LangGraph's serializer (see `OrchestratorState.data_context`'s docstring)
  - that's a real constraint, not a shortcut, so keep the docstring
    explaining why when a field can't just be the pydantic model directly.
- **Control flow**: prefer `bind_tools`/`tools_condition`/`ToolNode`,
  structured output (`with_structured_output`), and `InjectedState` over
  hand-rolled equivalents. A custom abstraction layer on top of these
  (e.g. the removed `chains.py` `RunnableLambda` wrappers) needs to earn
  its keep - if a node function calling the model directly is just as
  readable, drop the layer.

Custom code is still the right call when the framework genuinely has
nothing for it - e.g. `agents/orchestrator/history.py`'s rolling
summarization, or `agents/common/tools.py::flag_ambiguity`'s
report-don't-ask pattern for specialist ambiguity. The bar is "does this
custom piece do something the opinionated stack doesn't already offer",
not "could this be written some other way."

When in doubt, name the redundancy out loud (as with `FetchedDataset` vs.
the datasource tool's own dict, or the duplicate `ExecutionResult`/
`SandboxExecutionResult`) rather than quietly keeping both.
