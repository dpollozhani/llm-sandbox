from langchain_core.messages import AIMessage, HumanMessage
from pydantic import Field

from data_analyst.agents.datasource.chains import build_agent_chain
from data_analyst.clients.llm.factory import FakeToolCallingChatModel
from data_analyst.config.settings import Glossary, GlossaryEntry


class _RecordingLLM(FakeToolCallingChatModel):
    """Records the message list of every `_generate` call, so a test can
    check what the system prompt actually contained."""

    recorded_messages: list = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.recorded_messages.append(list(messages))
        return super()._generate(messages, stop=stop, run_manager=run_manager, **kwargs)


async def test_glossary_terms_are_injected_into_the_system_prompt():
    glossary = Glossary(
        terms=[GlossaryEntry(term="BRIC", definition="An item attribute, a global item code, not BRICS.")]
    )
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    chain = build_agent_chain(llm, tools=[], glossary=glossary)

    await chain.ainvoke([HumanMessage(content="hi")])

    system_prompt = llm.recorded_messages[0][0].content
    assert "Glossary:" in system_prompt
    assert "- BRIC: An item attribute, a global item code, not BRICS." in system_prompt


async def test_no_glossary_section_when_glossary_is_empty_or_absent():
    llm = _RecordingLLM(responses=[AIMessage(content="ok")])
    chain = build_agent_chain(llm, tools=[], glossary=Glossary())

    await chain.ainvoke([HumanMessage(content="hi")])

    assert "Glossary:" not in llm.recorded_messages[0][0].content

    llm2 = _RecordingLLM(responses=[AIMessage(content="ok")])
    chain2 = build_agent_chain(llm2, tools=[])  # glossary omitted entirely

    await chain2.ainvoke([HumanMessage(content="hi")])

    assert "Glossary:" not in llm2.recorded_messages[0][0].content
