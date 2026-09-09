from core.models import Goal, Task
from planner import Planner


def test_planner_turns_goal_into_a_plan():
    goal = Goal("Build a landing page")

    plan = Planner().create_plan(goal)

    assert plan.goal == goal
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "task-1"
    assert plan.tasks[0].description == "Build a landing page"


def test_planner_supports_custom_task_builder():
    def build(goal):
        return [
            Task("research", f"Research: {goal.description}"),
            Task("build", "Implement the result"),
        ]

    plan = Planner(build).create_plan(Goal("a website"))

    assert [task.id for task in plan.tasks] == ["research", "build"]


def test_planner_rejects_empty_goal():
    try:
        Planner().create_plan(Goal("  "))
    except ValueError as exc:
        assert "Goal description" in str(exc)
    else:
        raise AssertionError("Expected empty goal to fail")


def test_planner_rejects_duplicate_task_ids():
    def build(_goal):
        return [Task("same", "first"), Task("same", "second")]

    try:
        Planner(build).create_plan(Goal("test"))
    except ValueError as exc:
        assert "Duplicate task id" in str(exc)
    else:
        raise AssertionError("Expected duplicate task ids to fail")
