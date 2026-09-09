"""Specialized agent primitives for Daweling."""

from .base import AgentResult, BaseAgent
from .coding import CodingAgent
from .collaboration import AgentCollaborator, CollaborationResult, CollaborationStep
from .executor import AgentExecution, AgentExecutor
from .registry import AgentRegistry
from .research import ResearchAgent
from .router import AgentRouter

__all__ = [
    "AgentCollaborator",
    "AgentExecution",
    "AgentExecutor",
    "AgentResult",
    "AgentRouter",
    "BaseAgent",
    "CodingAgent",
    "CollaborationResult",
    "CollaborationStep",
    "AgentRegistry",
    "ResearchAgent",
]
