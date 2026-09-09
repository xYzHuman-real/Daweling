from memory import LearningEngine, MemoryStore


def test_learning_engine_retrieves_only_relevant_experiences(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.remember(
        "experience:one",
        {
            "goal": "Build Daweling memory",
            "success": True,
            "successful_tasks": 2,
            "failed_tasks": 0,
            "tools_used": ["echo"],
            "verification_reason": "verified",
        },
        category="experience",
    )
    store.remember(
        "experience:two",
        {
            "goal": "Write a photography caption",
            "success": True,
            "successful_tasks": 1,
            "failed_tasks": 0,
            "tools_used": ["writer"],
            "verification_reason": "verified",
        },
        category="experience",
    )

    context = LearningEngine(store).build("Build Daweling memory")

    assert len(context.experiences) == 1
    assert context.experiences[0].value["goal"] == "Build Daweling memory"
    assert "Build Daweling memory" in context.as_prompt_context()


def test_learning_engine_limits_results(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    for index in range(4):
        store.remember(
            f"experience:{index}",
            {"goal": "Build Daweling system", "success": bool(index % 2)},
            category="experience",
        )

    context = LearningEngine(store, max_results=2).build("Build Daweling system")

    assert len(context.experiences) == 2
