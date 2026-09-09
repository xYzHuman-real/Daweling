from core.models import Goal
from memory import MemoryStore
from memory.context import ContextEngine


def test_context_engine_retrieves_relevant_memory(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.remember("daweling.architecture", "execution-first AI system", category="architecture")
    store.remember("unrelated", "photography project", category="other")

    context = ContextEngine(store).build(Goal("Build Daweling architecture"))

    assert [entry.key for entry in context.memories] == ["daweling.architecture"]
    assert "execution-first AI system" in context.as_prompt_context()


def test_context_engine_preserves_goal_context(tmp_path):
    goal = Goal("Build an agent", {"project": "Daweling"})
    context = ContextEngine(MemoryStore(tmp_path / "memory.json")).build(goal)

    assert context.as_dict()["goal"] == "Build an agent"
    assert context.as_dict()["goal_context"]["project"] == "Daweling"
    assert context.memories == []
