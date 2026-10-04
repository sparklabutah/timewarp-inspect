from typing import Any

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageTool,
    ChatMessageUser,
    ModelName,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import ToolCall, ToolCallError

from timewarp.scorer import (
    llm_judge,
    parse_judge_verdict,
    submitted_answer,
    timewarp_scorer,
)

STRING_SPEC: dict[str, Any] = {
    "eval_types": ["string_match"],
    "reference_answers": {
        "must_include": ["biology"],
        "must_exclude": ["both", "neither"],
        "scope": "first_sentence",
        "fuzzy_match": "Biology",
    },
}

COMBINED_SPEC: dict[str, Any] = {
    "eval_types": ["string_match", "number_match"],
    "reference_answers": {
        "must_include": ["food and drug administration"],
        "number_match": {"value": 1906},
        "fuzzy_match": "Food and Drug Administration, 1906",
    },
}

JUDGE_SPEC: dict[str, Any] = {
    "eval_types": ["llm_judge"],
    "reference_answers": {
        "fuzzy_match": "(2) Ethiopian airliner crashes into Mediterranean Sea"
    },
}


def submit_messages(answer: str, call_id: str = "call_1") -> list[ChatMessage]:
    return [
        ChatMessageAssistant(
            content="",
            tool_calls=[
                ToolCall(id=call_id, function="submit", arguments={"answer": answer})
            ],
        ),
        ChatMessageTool(content=answer, tool_call_id=call_id, function="submit"),
    ]


def task_state(messages: list[ChatMessage], spec: dict[str, Any]) -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id="1_v1",
        epoch=1,
        input="question",
        messages=[ChatMessageUser(content="question"), *messages],
        metadata={"eval": spec, "intent": "question"},
    )


async def score(
    messages: list[ChatMessage], spec: dict[str, Any]
) -> tuple[Any, str | None]:
    result = await timewarp_scorer()(task_state(messages, spec), Target("unused"))
    assert result is not None
    return result.value, result.answer


def test_submitted_answer_is_the_last_submission() -> None:
    state = task_state(
        submit_messages("Physics", "call_1") + submit_messages("Biology", "call_2"),
        STRING_SPEC,
    )
    assert submitted_answer(state) == "Biology"


def test_submitted_answer_ignores_failed_submit_calls() -> None:
    failed = ChatMessageTool(
        content="",
        tool_call_id="call_2",
        function="submit",
        error=ToolCallError("parsing", "missing answer"),
    )
    state = task_state([*submit_messages("Biology"), failed], STRING_SPEC)
    assert submitted_answer(state) == "Biology"


def test_submitted_answer_none_without_submit() -> None:
    state = task_state(
        [ChatMessageAssistant(content="The answer is Biology.")], STRING_SPEC
    )
    assert submitted_answer(state) is None


async def test_scorer_accepts_correct_answer() -> None:
    value, answer = await score(
        submit_messages("Biology. Biophysics is listed there."), STRING_SPEC
    )
    assert value == CORRECT
    assert answer == "Biology. Biophysics is listed there."


async def test_scorer_rejects_wrong_answer() -> None:
    value, _ = await score(submit_messages("Both articles mention it."), STRING_SPEC)
    assert value == INCORRECT


async def test_scorer_ignores_unsubmitted_text() -> None:
    """Text the agent never sent with submit() must not be scored, as upstream."""
    value, answer = await score([ChatMessageAssistant(content="Biology")], STRING_SPEC)
    assert value == INCORRECT
    assert answer is None


async def test_scorer_requires_every_verifier() -> None:
    passing, _ = await score(
        submit_messages("The Food and Drug Administration was formed in 1906."),
        COMBINED_SPEC,
    )
    failing, _ = await score(
        submit_messages("The Food and Drug Administration was formed in 1907."),
        COMBINED_SPEC,
    )
    assert passing == CORRECT
    assert failing == INCORRECT


async def test_scorer_rejects_unknown_eval_type() -> None:
    spec = {"eval_types": ["program_html"], "reference_answers": {}}
    with pytest.raises(ValueError, match="program_html"):
        await score(submit_messages("anything"), spec)


@pytest.mark.parametrize(
    ("response", "passed"),
    [
        ("correct", True),
        ("Correct", True),
        ("incorrect", False),
        ("partially correct", False),
        ("The answer is correct.", True),
        ("This is partially correct.", False),
        ("unsure", False),
    ],
)
def test_parse_judge_verdict(response: str, passed: bool) -> None:
    assert parse_judge_verdict(response) is passed


@solver
def scripted_submission(answer: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.messages.extend(submit_messages(answer))
        return state

    return solve


@pytest.mark.parametrize(
    ("verdict", "expected"), [("correct", CORRECT), ("partially correct", INCORRECT)]
)
def test_llm_judge_uses_grader_role(verdict: str, expected: str) -> None:
    grader = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(model="mockllm/model", content=verdict)
        ],
    )
    task = Task(
        dataset=[
            Sample(
                id="32_v1",
                input="Which crash was deadlier?",
                target="(2) Ethiopian airliner crashes into Mediterranean Sea",
                metadata={
                    "eval": JUDGE_SPEC,
                    "intent": "Which crash was deadlier?",
                    "ui_version": 1,
                    "category": "news",
                },
            )
        ],
        solver=scripted_submission("The Ethiopian airliner crash."),
        scorer=timewarp_scorer(),
    )
    [log] = eval(task, model="mockllm/model", model_roles={"grader": grader})

    assert log.status == "success"
    assert log.samples is not None
    sample_score = log.samples[0].scores
    assert sample_score is not None
    assert sample_score["timewarp_scorer"].value == expected


async def test_llm_judge_na_reference_needs_no_grader_call() -> None:
    grader = get_model("mockllm/model", custom_outputs=[])
    assert await llm_judge(grader, " n/a ", {"fuzzy_match": "N/A"}, "q")
    assert not await llm_judge(grader, "Biology", {"fuzzy_match": "N/A"}, "q")


async def test_llm_judge_passes_if_any_reference_matches() -> None:
    grader = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.from_content(model="mockllm/model", content="incorrect"),
            ModelOutput.from_content(model="mockllm/model", content="correct"),
        ],
    )
    assert await llm_judge(grader, "B", {"fuzzy_match": ["A", "B"]}, "q")


async def test_llm_judge_requires_fuzzy_match() -> None:
    with pytest.raises(ValueError, match="fuzzy_match"):
        await llm_judge(get_model("mockllm/model"), "B", {}, "q")
