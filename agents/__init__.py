"""Specialized agent primitives for Daweling."""

from .base import AgentResult, BaseAgent
from .coding import CodingAgent
from .executor import AgentExecution, AgentExecutor
from .registry import AgentRegistry
from .research import ResearchAgent
from .router import AgentRouter

__all__ = [
    "AgentExecution",
    "AgentExecutor",
    "AgentResult",
    "AgentRouter",
    "BaseAgent",
    "CodingAgent",
    "AgentRegistry",
    "ResearchAgent",
]
