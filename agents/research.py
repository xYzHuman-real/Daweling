"""Research agent foundation for Daweling."""

from typing import Any

from models import ModelMessage, ModelProvider

from .base import AgentResult, BaseAgent


class ResearchAgent(BaseAgent):
    """Use a model to structure research work; external search tools come later."""

    name = "research"
    description = "Break research questions into focused, evidence-oriented work."
    capabilities = ("research", "summarization", "evidence")

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        if not task.strip():
            return AgentResult.fail("Research task cannot be empty")
        response = self.provider.generate([
            ModelMessage(
                role="system",
                content=(
                    "You are Daweling's research agent. Do not claim to have browsed "
                    "or verified external sources unless tools actually provide them. "
                    "Return a concise research plan or synthesis based only on supplied context."
                ),
            ),
            ModelMessage(role="user", content=task.strip()),
        ])
        return AgentResult.ok(response.content, model=response.model, agent=self.name)
