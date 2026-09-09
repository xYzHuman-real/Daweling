"""Public application service for running Daweling end to end."""

from core.models import Goal
from core.runtime import Runtime
from memory import ExperienceRecorder, MemoryStore
from memory.context import ContextEngine
from models import ModelProvider
from orchestrator import ExecutionResult, Orchestrator
from planner.model_action_builder import ModelActionBuilder
from planner.model_planner import ModelPlanner
from tools.registry import ToolRegistry


class Daweling:
    """Run goals through model, memory, planning, tools, verification, and learning."""

    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.memory = memory or MemoryStore()
        self.context_engine = ContextEngine(self.memory)
        self.experience_recorder = ExperienceRecorder(self.memory)
        self.planner = ModelPlanner(provider)
        self.action_builder = ModelActionBuilder(provider, self.registry)
        self.orchestrator = Orchestrator(
            planner=self.planner,
            runtime=Runtime(self.registry),
        )

    def run(self, goal: Goal) -> ExecutionResult:
        """Execute one goal and record a compact experience for future learning."""
        context = self.context_engine.build(goal)
        plan = self.planner.create_plan(goal, context=context)
        actions = self.action_builder.build_actions(plan)
        self.orchestrator._validate_actions(plan, actions)
        observations = self.orchestrator.runtime.execute(plan, actions)
        verifications = [
            self.orchestrator.runtime.verify(observation)
            for observation in observations
        ]
        result = ExecutionResult(
            plan=plan,
            observations=observations,
            verifications=verifications,
        )

        self.experience_recorder.record(
            goal=goal.description,
            success=result.success,
            verifications=verifications,
            task_count=len(plan.tasks),
            successful_tasks=sum(observation.success for observation in observations),
            failed_tasks=sum(not observation.success for observation in observations),
            tools_used=[action.tool for action in actions],
        )
        return result
