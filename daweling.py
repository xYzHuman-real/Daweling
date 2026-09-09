"""Public application service for running Daweling end to end."""

from agents import AgentExecutor, AgentRegistry, AgentExecution, AgentRouter, CodingAgent, ResearchAgent
from core.models import Goal
from core.policy import ApprovalPolicy
from core.runtime import Runtime
from memory import ExperienceRecorder, MemoryStore
from memory.context import ContextEngine
from models import ModelProvider
from orchestrator import ExecutionResult, Orchestrator
from planner.model_action_builder import ModelActionBuilder
from planner.model_planner import ModelPlanner
from tools import CodeRunnerTool, WebSearchTool
from tools.config import CapabilityConfig
from tools.registry import ToolRegistry


class Daweling:
    """Run goals through model, agents, memory, planning, tools, verification, and learning."""

    def __init__(
        self,
        provider: ModelProvider,
        registry: ToolRegistry | None = None,
        memory: MemoryStore | None = None,
        agent_registry: AgentRegistry | None = None,
        approval_policy: ApprovalPolicy | None = None,
        capability_config: CapabilityConfig | None = None,
    ) -> None:
        self.registry = registry or ToolRegistry()
        self.memory = memory or MemoryStore()
        self.context_engine = ContextEngine(self.memory)
        self.experience_recorder = ExperienceRecorder(self.memory)
        self.planner = ModelPlanner(provider)
        self.action_builder = ModelActionBuilder(provider, self.registry)

        self.agent_registry = agent_registry or AgentRegistry([
            ResearchAgent(provider),
            CodingAgent(provider),
        ])
        self.agent_router = AgentRouter(self.agent_registry)
        self.agent_executor = AgentExecutor(self.agent_router)

        self._configure_capabilities(capability_config or CapabilityConfig.from_env())
        self.orchestrator = Orchestrator(
            planner=self.planner,
            runtime=Runtime(self.registry, approval_policy=approval_policy),
        )

    def _configure_capabilities(self, config: CapabilityConfig) -> None:
        """Register only capabilities whose external endpoints are explicitly configured."""
        if config.web_search_url and "web_search" not in self.registry:
            self.registry.register(
                WebSearchTool.from_http(
                    config.web_search_url,
                    api_key=config.web_search_api_key,
                    timeout=config.web_search_timeout,
                )
            )
        if config.code_runner_url and "code_runner" not in self.registry:
            self.registry.register(
                CodeRunnerTool.from_http(
                    config.code_runner_url,
                    api_key=config.code_runner_api_key,
                    timeout=config.code_runner_timeout,
                )
            )

    def run(self, goal: Goal) -> ExecutionResult:
        """Execute Goal → Plan → Agent → Action → Tool → Verify → Learn."""
        context = self.context_engine.build(goal)
        plan = self.planner.create_plan(goal, context=context)

        agent_executions = self.agent_executor.execute(
            plan,
            context=context.as_dict(),
        )
        agent_work = self.agent_executor.as_action_context(agent_executions)

        actions = self.action_builder.build_actions(plan, agent_context=agent_work)
        self.orchestrator.validate_actions(plan, actions)
        observations = self.orchestrator.runtime.execute(plan, actions)
        verifications = [
            self.orchestrator.runtime.verify(observation)
            for observation in observations
        ]
        result = ExecutionResult(
            plan=plan,
            observations=observations,
            verifications=verifications,
            agent_work=agent_work,
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
