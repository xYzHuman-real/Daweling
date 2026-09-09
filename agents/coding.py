"""Coding agent foundation for Daweling."""

from typing import Any

from models import ModelMessage, ModelProvider

from .base import AgentResult, BaseAgent


class CodingAgent(BaseAgent):
    """Plan and explain coding work while execution remains tool-controlled."""

    name = "coding"
    description = "Analyze software tasks and produce implementation-oriented guidance."
    capabilities = ("coding", "debugging", "software-engineering")

    def __init__(self, provider: ModelProvider) -> None:
        self.provider = provider

    def run(self, task: str, context: dict[str, Any] | None = None) -> AgentResult:
        if not task.strip():
            return AgentResult.fail("Coding task cannot be empty")
        response = self.provider.generate([
            ModelMessage(
                role="system",
                content=(
                    "You are Daweling's coding agent. Analyze the requested software task "
                    "and provide implementation guidance. Never claim code was executed "
                    "or tested unless an execution tool actually reports that result."
                ),
            ),
            ModelMessage(role="user", content=task.strip()),
        ])
        return AgentResult.ok(response.content, model=response.model, agent=self.name)
