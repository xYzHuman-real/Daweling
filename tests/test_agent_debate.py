from agents.base import AgentResult, BaseAgent
from agents.debate import AgentDebate


class Reviewer(BaseAgent):
    name = "reviewer"
    description = "Peer reviewer"
    capabilities = ("review",)

    def __init__(self, output):
        self.output = output

    def run(self, task, context=None):
        assert "work_to_review" in (context or {})
        return AgentResult.ok(self.output)


def test_debate_accepts_work_when_all_reviewers_approve():
    debate = AgentDebate([
        Reviewer({"approved": True, "critique": "Looks correct."}),
        Reviewer({"approved": True, "critique": "Evidence is sufficient."}),
    ])
    result = debate.review("answer", task="verify answer")
    assert result.accepted
    assert len(result.reviews) == 2


def test_debate_rejects_work_when_a_reviewer_finds_a_problem():
    debate = AgentDebate([
        Reviewer({"approved": True, "critique": "Fine."}),
        Reviewer({"approved": False, "critique": "Missing verification."}),
    ])
    result = debate.review("answer", task="verify answer")
    assert not result.accepted
    assert result.reviews[1].critique == "Missing verification."


def test_debate_requires_structured_reviewer_output():
    debate = AgentDebate([Reviewer("looks good")])
    result = debate.review("answer", task="verify answer")
    assert not result.accepted
