"""Specialized agent primitives for Daweling."""

from .base import AgentResult, BaseAgent
from .coding import CodingAgent
from .collaboration import AgentCollaborator, CollaborationResult, CollaborationStep
from .debate import AgentDebate, DebateResult, ReviewDecision
from .executor import AgentExecution, AgentExecutor
from .pipeline import AgentPipeline, AgentPipelineResult
from .registry import AgentRegistry
from .research import ResearchAgent
from .router import AgentRouter

__all__ = [
    "AgentCollaborator",
    "AgentDebate",
    "AgentExecution",
    "AgentExecutor",
    "AgentPipeline",
    "AgentPipelineResult",
    "AgentResult",
    "AgentRouter",
    "BaseAgent",
    "CodingAgent",
    "CollaborationResult",
    "CollaborationStep",
    "DebateResult",
    "ReviewDecision",
    "AgentRegistry",
    "ResearchAgent",
]
