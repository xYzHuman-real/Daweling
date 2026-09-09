from core.models import Goal, Plan, Task
from core.reasoning import ReasoningEngine
from models import ModelProvider, ModelResponse
from orchestrator.intelligence import IntelligenceCore
from planner.model_planner import ModelPlanner


class FakeProvider(ModelProvider):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.messages = []

    def generate(self, messages, **kwargs):
        self.messages.append(messages)
        return ModelResponse(next(self.responses), model="fake")


def test_intelligence_core_reasons_before_planning():
    provider = FakeProvider([
        '{"understanding":"understand goal","strategy":"research first","steps":[{"purpose":"scope","conclusion":"define evidence needed"}],"uncertainties":[]}',
        '{"tasks":[{"id":"research","description":"Research the evidence"}]}'
    ])
    core = IntelligenceCore(ReasoningEngine(provider), ModelPlanner(provider))

    result = core.prepare(Goal("Create a research brief"))

    assert result.reasoning.strategy == "research first"
    assert result.plan.tasks == [Task("research", "Research the evidence")]
    assert "research first" in provider.messages[1][-1].content


def test_intelligence_core_preserves_goal_context_for_planning():
    provider = FakeProvider([
        '{"understanding":"u","strategy":"s","steps":[{"purpose":"p","conclusion":"c"}],"uncertainties":[]}',
        '{"tasks":[{"id":"task-1","description":"Do the task"}]}'
    ])
    core = IntelligenceCore(ReasoningEngine(provider), ModelPlanner(provider))
    result = core.prepare(Goal("test", {"priority": "high"}))

    assert result.plan.goal.context["goal_context"] == {"priority": "high"}
    assert result.plan.goal.context["reasoning"]["strategy"] == "s"
