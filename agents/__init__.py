"""Specialized agent primitives for Daweling."""

from .base import AgentResult, BaseAgent
from .coding import CodingAgent
from .registry import AgentRegistry
from .research import ResearchAgent

__all__ = [
    "AgentResult",
    "BaseAgent",
    "CodingAgent",
    "AgentRegistry",
    "ResearchAgent",
]
