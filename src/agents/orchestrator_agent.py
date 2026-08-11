from agents.base import Agent
from agents.synthesis_agent import SynthesisAgent
from agents.tools.agent_tool import AgentTool
from agents.tools.base import ToolRegistry
from llm.base import LLMClient
from rag.source_filter import SourceExclusions
from rag.types import RetrievalFilters
from store.base import VectorStore

_MAX_STEPS = 3

_DECISION_ROLE = (
    "You are a conversational assistant for a software team. Hold a normal, "
    "helpful conversation. When answering needs facts about the team's own "
    "project — its code, docs, retros, or tickets — call the knowledge sub-agent "
    "to look them up instead of guessing; you may call it more than once for a "
    "question with several distinct parts. For greetings, small talk, or general "
    "questions that need no project-specific facts, just reply directly without "
    "calling anything."
)

_ANSWER_SYSTEM = """\
You are SprintStart's assistant for software teams.
When your sub-agents gathered context, base your answer on it. When nothing was
gathered, respond naturally as a helpful assistant.
Be concise and precise. Use markdown formatting where appropriate.
"""


class OrchestratorAgent(Agent):
    def __init__(
        self,
        llm: LLMClient,
        store: VectorStore,
        exclusions: SourceExclusions = SourceExclusions(),
        filters: RetrievalFilters | None = None,
    ) -> None:
        tools = ToolRegistry(
            [
                AgentTool(SynthesisAgent(llm, store, exclusions, filters)),
            ]
        )
        super().__init__(
            name="orchestrator",
            description="Chats with the developer and consults sub-agents on demand.",
            llm=llm,
            tools=tools,
            decision_role=_DECISION_ROLE,
            answer_system=_ANSWER_SYSTEM,
            max_steps=_MAX_STEPS,
        )
