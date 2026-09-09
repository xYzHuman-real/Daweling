"""Persistent, provider-independent memory primitives for Daweling."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class MemoryEntry:
    """A durable piece of context associated with a project or session."""

    key: str
    value: Any
    category: str = "general"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class MemoryStore:
    """Simple JSON-backed memory store with deterministic retrieval."""

    def __init__(self, path: str | Path = "data/memory.json") -> None:
        self.path = Path(path)
        self._entries: list[MemoryEntry] = []
        self._load()

    def remember(self, key: str, value: Any, category: str = "general") -> MemoryEntry:
        """Store or replace a memory by key."""
        key = key.strip()
        category = category.strip() or "general"
        if not key:
            raise ValueError("Memory key cannot be empty")

        entry = MemoryEntry(key=key, value=value, category=category)
        self._entries = [item for item in self._entries if item.key != key]
        self._entries.append(entry)
        self._save()
        return entry

    def recall(self, key: str) -> MemoryEntry | None:
        """Retrieve one exact memory by key."""
        for entry in reversed(self._entries):
            if entry.key == key:
                return entry
        return None

    def search(self, query: str, category: str | None = None) -> list[MemoryEntry]:
        """Return memories whose key or textual value contains every query term."""
        terms = [term.lower() for term in query.split() if term.strip()]
        if not terms:
            return []

        matches: list[MemoryEntry] = []
        for entry in reversed(self._entries):
            if category and entry.category != category:
                continue
            haystack = f"{entry.key} {entry.value}".lower()
            if all(term in haystack for term in terms):
                matches.append(entry)
        return matches

    def all(self) -> list[MemoryEntry]:
        """Return all memories, newest first."""
        return list(reversed(self._entries))

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to load memory store: {self.path}") from exc
        if not isinstance(raw, list):
            raise ValueError("Memory store must contain a JSON array")
        self._entries = [MemoryEntry(**item) for item in raw if isinstance(item, dict)]

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [entry.__dict__ for entry in self._entries]
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
