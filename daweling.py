"""Public application service for running Daweling end to end."""

from agents import AgentExecutor, AgentRegistry, AgentRouter, CodingAgent, ResearchAgent
from core.models import Action, Goal
from core.policy import ApprovalPolicy
from core.recovery import RecoveryCallback
from core.runtime import Runtime
from memory import ExperienceRecorder, MemoryStore
from memory.context import ContextEngine
from models import ModelProvider
from orchestrator import DecisionDrivenLoop, ExecutionResult, NextStep, Orchestrator
from planner import AdaptivePlanner
from planner.model_action_builder import ModelActionBuilder
from planner.model_planner import ModelPlanner
from tools import CodeRunnerTool, WebSearchTool
from tools.config import CapabilityConfig
from tools.registry import ToolRegistry


class Daweling:
    """Run goals through model, agents, memory, planning, tools, verification, and learning."""

    def __init__(self, provider: ModelProvider, registry: ToolRegistry | None = None,
                 memory: MemoryStore | None = None, agent_registry: AgentRegistry | None = None,
                 approval_policy: ApprovalPolicy | None = None,
                 capability_config: CapabilityConfig | None = None, max_recovery_attempts: int = 2,
                 max_replan_rounds: int = 2) -> None:
        self.provider = provider
        self.registry = registry or ToolRegistry()
        self.memory = memory or MemoryStore()
        self.context_engine = ContextEngine(self.memory)
        self.experience_recorder = ExperienceRecorder(self.memory)
        self.planner = ModelPlanner(provider)
        self.adaptive_planner = AdaptivePlanner(provider)
        self.action_builder = ModelActionBuilder(provider, self.registry)
        self.agent_registry = agent_registry or AgentRegistry([ResearchAgent(provider), CodingAgent(provider)])
        self.agent_router = AgentRouter(self.agent_registry)
        self.agent_executor = AgentExecutor(self.agent_router)
        self._configure_capabilities(capability_config or CapabilityConfig.from_env())
        runtime = Runtime(self.registry, approval_policy=approval_policy)
        self.orchestrator = Orchestrator(planner=self.planner, runtime=runtime)
        self.loop = DecisionDrivenLoop(
            runtime,
            planner=self.planner,
            max_recovery_attempts=max_recovery_attempts,
            max_replan_rounds=max_replan_rounds,
        )
        self.max_recovery_attempts = max_recovery_attempts
        self.max_replan_rounds = max_replan_rounds

    def _configure_capabilities(self, config: CapabilityConfig) -> None:
        if config.web_search_url and "web_search" not in self.registry:
            self.registry.register(WebSearchTool.from_http(config.web_search_url, api_key=config.web_search_api_key, timeout=config.web_search_timeout))
        if config.code_runner_url and "code_runner" not in self.registry:
            self.registry.register(CodeRunnerTool.from_http(config.code_runner_url, api_key=config.code_runner_api_key, timeout=config.code_runner_timeout))

    def _prepare(self, goal: Goal):
        context = self.context_engine.build(goal)
        plan = self.planner.create_plan(goal, context=context)
        agent_executions = self.agent_executor.execute(plan, context=context.as_dict())
        agent_work = self.agent_executor.as_action_context(agent_executions)
        actions = self.action_builder.build_actions(plan, agent_context=agent_work)
        self.orchestrator.validate_actions(plan, actions)
        return plan, actions, agent_work

    def _record_result(self, goal, result, actions, diagnoses=()):
        self.experience_recorder.record(
            goal=goal.description, success=result.success, verifications=result.verifications,
            task_count=len(result.plan.tasks), successful_tasks=sum(o.success for o in result.observations),
            failed_tasks=sum(not o.success for o in result.observations),
            tools_used=[a.tool for a in actions], recovery_diagnoses=list(diagnoses),
            recovery_attempts=len(diagnoses),
        )
        return result

    def run(self, goal: Goal) -> ExecutionResult:
        plan, actions, agent_work = self._prepare(goal)
        observations = self.orchestrator.runtime.execute(plan, actions)
        verifications = [self.orchestrator.runtime.verify(o) for o in observations]
        return self._record_result(goal, ExecutionResult(plan, observations, verifications, agent_work), actions)

    def run_with_recovery(self, goal: Goal, recover: RecoveryCallback | None = None) -> ExecutionResult:
        """Run with bounded model-guided recovery and adaptive replanning."""
        plan, actions, agent_work = self._prepare(goal)
        current_actions = actions
        observations = []
        verifications = []
        executed_actions = list(actions)
        diagnoses = []
        recovery_attempts = 0
        replan_rounds = 0

        while True:
            observations = self.orchestrator.runtime.execute(plan, current_actions)
            verifications = [self.orchestrator.runtime.verify(o) for o in observations]
            if all(item.valid for item in verifications) and verifications:
                result = ExecutionResult(plan, observations, verifications, agent_work)
                return self._record_result(goal, result, executed_actions, diagnoses)

            failed_index = next((i for i, item in enumerate(verifications) if not item.valid), None)
            if failed_index is None:
                result = ExecutionResult(plan, observations, verifications, agent_work)
                return self._record_result(goal, result, executed_actions, diagnoses)

            failed_action = current_actions[failed_index]
            failed_observation = observations[failed_index]
            if recovery_attempts < self.max_recovery_attempts:
                if recover is not None:
                    replacement = recover(failed_action, failed_observation, recovery_attempts + 1)
                    diagnosis = "Custom recovery callback"
                else:
                    decision = self.action_builder.build_recovery_action_with_diagnosis(
                        plan, failed_action, failed_observation, recovery_attempts + 1
                    )
                    replacement = decision.action
                    diagnosis = decision.diagnosis
                recovery_attempts += 1
                diagnoses.append(diagnosis)
                if replacement is not None:
                    current_actions = list(current_actions)
                    current_actions[failed_index] = replacement
                    executed_actions.append(replacement)
                    continue

            if replan_rounds < self.max_replan_rounds:
                replanned = self.adaptive_planner.replan(
                    plan, observations, verifications, current_actions
                )
                plan = replanned.plan
                replan_rounds += 1
                recovery_attempts = 0
                current_actions = self.action_builder.build_actions(plan, agent_context=agent_work)
                self.orchestrator.validate_actions(plan, current_actions)
                executed_actions.extend(current_actions)
                continue

            result = ExecutionResult(plan, observations, verifications, agent_work)
            return self._record_result(goal, result, executed_actions, diagnoses)

    def run_autonomous(self, goal: Goal) -> ExecutionResult:
        """Run the model-guided recovery/replanning control loop end to end."""
        return self.run_with_recovery(goal)
