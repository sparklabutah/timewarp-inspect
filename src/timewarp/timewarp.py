"""TimeWarp: Evaluating Web Agents by Revisiting the Past.

Md Farhan Ishmam and Kenneth Marino
https://arxiv.org/abs/2603.04949

Agents answer questions and complete orders on three locally served websites
(Wiki, News and Shop), each rendered in six UI versions from different eras of
the web. The same 231 goals are posed in every version.
"""

from collections.abc import Sequence
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.agent import Agent
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import Scorer
from inspect_ai.solver import Solver

from timewarp.agent import DEFAULT_MAX_RETRIES, browser_agent
from timewarp.dataset import CATEGORIES, UI_VERSIONS, Split, timewarp_dataset
from timewarp.scorer import timewarp_scorer

COMPOSE_FILE = Path(__file__).parent / "sandbox" / "compose.yaml"

# Maximum agent steps per episode in the paper and in BrowserGym's `timewarp` benchmark.
DEFAULT_MAX_STEPS = 30


@task
def timewarp(
    split: Split = "test",
    ui_versions: int | Sequence[int] | None = None,
    categories: str | Sequence[str] | None = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    screenshots: bool = False,
    solver: Solver | Agent | None = None,
    scorer: Scorer | None = None,
) -> Task:
    """TimeWarp web-agent benchmark.

    Args:
        split: "test" (103 goals, used for the paper's results), "train"
            (128 goals) or "all" (231 goals).
        ui_versions: UI versions to include, from 1 (oldest era) to 6 (minimal
            control). Defaults to all six.
        categories: Goal categories to include: "wiki", "news", "shop" and/or
            "multi" (goals spanning several sites). Defaults to all four.
        max_steps: Maximum number of agent steps, including the final answer,
            as in the paper. A step is one browser action; turns without a
            usable tool call are retried within the step (up to 4 times, as in
            AgentLab). For any solver, the task also caps model turns at
            `max_steps * 5`.
        screenshots: Give the default agent the `browser_take_screenshot` tool,
            for vision-capable models.
        solver: Solver or agent to run instead of `browser_agent()`. Its final
            answer must remain in the transcript as the result of a tool named
            `submit` (with `react()`, pass `AgentSubmit(keep_in_messages=True)`).
        scorer: Scorer to use instead of the TimeWarp verifiers.
    """
    if split not in ("test", "train", "all"):
        raise ValueError(f"split must be 'test', 'train' or 'all', got {split!r}")

    versions = _as_versions(ui_versions)
    invalid_versions = [v for v in versions if v not in UI_VERSIONS]
    if invalid_versions or not versions:
        raise ValueError(f"ui_versions must be values from 1 to 6, got {versions}")

    selected_categories = _as_categories(categories)
    invalid_categories = [c for c in selected_categories if c not in CATEGORIES]
    if invalid_categories or not selected_categories:
        raise ValueError(
            f"categories must be values from {list(CATEGORIES)}, got {selected_categories}"
        )

    if max_steps < 1:
        raise ValueError(f"max_steps must be at least 1, got {max_steps}")

    return Task(
        dataset=timewarp_dataset(split, versions, selected_categories),
        solver=solver or browser_agent(max_steps=max_steps, screenshots=screenshots),
        scorer=scorer or timewarp_scorer(),
        sandbox=("docker", str(COMPOSE_FILE)),
        turn_limit=max_steps * (DEFAULT_MAX_RETRIES + 1),
        config=GenerateConfig(parallel_tool_calls=False),
        version="1-A",
    )


def _as_versions(value: int | Sequence[int] | None) -> list[int]:
    if value is None:
        return list(UI_VERSIONS)
    return [value] if isinstance(value, int) else list(value)


def _as_categories(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return list(CATEGORIES)
    return [value] if isinstance(value, str) else list(value)
