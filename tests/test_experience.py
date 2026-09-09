from core.models import VerificationResult
from memory import ExperienceRecorder, MemoryStore


def test_experience_recorder_persists_successful_workflow(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    recorder = ExperienceRecorder(store)

    experience = recorder.record(
        goal="Build Daweling memory",
        success=True,
        verifications=[VerificationResult.passed("Everything verified")],
        task_count=2,
        successful_tasks=2,
        failed_tasks=0,
        tools_used=["echo", "echo"],
    )

    stored = store.search("experience", category="experience")
    assert len(stored) == 1
    assert stored[0].value["success"] is True
    assert experience.tools_used == ["echo"]


def test_experience_recorder_persists_failure_without_raw_payloads(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    recorder = ExperienceRecorder(store)

    recorder.record(
        goal="Run a failed workflow",
        success=False,
        verifications=[VerificationResult.failed("Tool execution failed")],
        task_count=1,
        successful_tasks=0,
        failed_tasks=1,
        tools_used=["risky_tool"],
    )

    entry = store.search("failed workflow", category="experience")[0]
    assert entry.value["success"] is False
    assert entry.value["failed_tasks"] == 1
    assert "input" not in entry.value
    assert "output" not in entry.value
