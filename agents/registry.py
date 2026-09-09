"""Registry for specialized Daweling agents."""

from .base import BaseAgent


class AgentRegistry:
    """Register and discover specialized agents by name."""

    def __init__(self, agents: list[BaseAgent] | None = None) -> None:
        self._agents: dict[str, BaseAgent] = {}
        for agent in agents or []:
            self.register(agent)

    def register(self, agent: BaseAgent) -> None:
        name = agent.name.strip()
        if not name:
            raise ValueError("Agent name cannot be empty")
        if name in self._agents:
            raise ValueError(f"Agent already registered: {name}")
        self._agents[name] = agent

    def get(self, name: str) -> BaseAgent | None:
        return self._agents.get(name)

    def require(self, name: str) -> BaseAgent:
        agent = self.get(name)
        if agent is None:
            raise KeyError(f"Agent not found: {name}")
        return agent

    def list(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents)
