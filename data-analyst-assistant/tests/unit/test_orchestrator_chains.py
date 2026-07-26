from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from pydantic import Field

from data_analyst.agents.orchestrator.chains import build_clarify_chain, build_respond_chain, build_supervisor_chain
from data_analyst.clients.llm.factory import FakeToolCallingChatModel
from data_analyst.config.settings import Glossary, GlossaryEntry

_GLOSSARY = Glossary(terms=[GlossaryEntry(term="BRIC", definition="An item attribute, not BRICS.")])


class _RecordingLLM(FakeToolCallingChatModel):
    """Records the message list of every `_generate` call."""

    recorded_messages: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recorded_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


class _RecordingStructuredLLM(FakeToolCallingChatModel):
    """Like `_RecordingLLM`, but for the `with_structured_output` path
    (`build_supervisor_chain`/`build_clarify_chain`) - the base fake's own
    `with_structured_output` stand-in ignores its input entirely, so it
    can't be used to check what prompt a structured-output call actually
    received."""

    recorded_messages: list = Field(default_factory=list)

    def with_structured_output(self, schema, **kwargs):
        def _invoke(messages, *_args, **_kwargs):
            self.recorded_messages.append(list(messages))
            return schema()

        return RunnableLambda(_invoke)


async def test_respond_chain_injects_glossary():
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    chain = build_respond_chain(llm, glossary=_GLOSSARY)

    await chain.ainvoke([HumanMessage(content="hi")])

    assert "- BRIC: An item attribute, not BRICS." in llm.recorded_messages[0][0].content


async def test_respond_chain_has_no_glossary_section_when_absent():
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    chain = build_respond_chain(llm)

    await chain.ainvoke([HumanMessage(content="hi")])

    assert "Glossary:" not in llm.recorded_messages[0][0].content


async def test_supervisor_chain_injects_glossary():
    """The supervisor sees raw user language before ever delegating - a
    term it doesn't recognize can misroute, or trigger an unnecessary
    upfront clarifying question, just as easily as it can confuse the
    datasource agent."""
    llm = _RecordingStructuredLLM(responses=[])
    chain = build_supervisor_chain(llm, glossary=_GLOSSARY)

    await chain.ainvoke({"messages": [HumanMessage(content="top 10 by BRIC")], "data_context": None})

    assert "- BRIC: An item attribute, not BRICS." in llm.recorded_messages[0][0].content


async def test_clarify_chain_injects_glossary():
    llm = _RecordingStructuredLLM(responses=[])
    chain = build_clarify_chain(llm, glossary=_GLOSSARY)

    await chain.ainvoke([HumanMessage(content="hi")])

    assert "- BRIC: An item attribute, not BRICS." in llm.recorded_messages[0][0].content
