"""Public application service for running Daweling end to end."""

from core.models import Goal
from core.runtime import Runtime
from models import ModelProvider
from orchestrator import ExecutionResult, Orchestrator
from planner.model_action_builder import ModelActionBuilder
from planner.model_planner import ModelPlanner
from tools.registry import ToolRegistry


class Daweling:
    """Run a goal through Daweling's model, planning, tools, and verification layers.

    The model provider is injected so Daweling can use an external provider today
    and a first-party model later without changing the application interface.
    """

    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.planner = ModelPlanner(provider)
        self.action_builder = ModelActionBuilder(provider, self.registry)
        self.orchestrator = Orchestrator(
            planner=self.planner,
            runtime=Runtime(self.registry),
        )

    def run(self, goal: Goal) -> ExecutionResult:
        """Execute one goal through plan → actions → tools → verification."""
        return self.orchestrator.run(goal, self.action_builder.build_actions)
