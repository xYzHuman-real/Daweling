from memory import MemoryStore


def test_memory_store_persists_and_recalls(tmp_path):
    path = tmp_path / "memory.json"
    store = MemoryStore(path)
    store.remember("project.name", "Daweling", category="project")

    reloaded = MemoryStore(path)
    entry = reloaded.recall("project.name")

    assert entry is not None
    assert entry.value == "Daweling"
    assert entry.category == "project"


def test_memory_search_matches_all_terms(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.remember("preferred_model", "future Daweling model", category="architecture")
    store.remember("launch_goal", "build a reliable AI system", category="product")

    matches = store.search("Daweling model", category="architecture")

    assert [entry.key for entry in matches] == ["preferred_model"]


def test_memory_replaces_existing_key(tmp_path):
    store = MemoryStore(tmp_path / "memory.json")
    store.remember("status", "draft")
    store.remember("status", "active")

    assert store.recall("status").value == "active"
    assert len(store.all()) == 1
