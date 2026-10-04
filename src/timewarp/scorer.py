"""TimeWarp scorer: the upstream verifiers applied to the agent's submitted answer."""

from collections.abc import Mapping
from typing import Any

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    GenerateConfig,
    Model,
    get_model,
)
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    grouped,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from timewarp.verifiers import DETERMINISTIC_VERIFIERS

# Judge used for the paper's LLM-judged goals (Table 17 of arXiv:2603.04949v3).
DEFAULT_GRADER_MODEL = "openai/gpt-5.1-2025-11-13"

SUBMIT_TOOL = "submit"

# Verbatim from `llm_fuzzy_match` in the upstream `evaluators.py`.
JUDGE_PROMPT = (
    """Help a teacher grade the answer of a student given a question. Keep in mind that the student may use different phrasing or wording to answer the question. The goal is to evaluate whether the answer is semantically equivalent to the reference answer.
Input:
- question: {question}
- reference answer: {reference}
- student answer: {answer}

Special Sequence: The string 'N/A' that you see is a special sequence that means 'not achievable'

Output: You must respond with EXACTLY one of the following words (nothing else):
1) 'correct': if the answer is semantically equivalent to the reference."""
    # Upstream has a trailing space on this line; it is kept so the prompt is identical.
    " \n"
    """   - Numeric values must match exactly (including units, signs, and scale) unless the question/reference clearly allows an approximation or rounding.
   - If an estimate is allowed, the student must still be reasonably close and not contradict the reference.
   - Ordered lists/steps/rankings must match exactly in both the element values and order.
   - Unordered lists/sets must contain the same element values; the order of the elements does not matter.
   - Extra information is allowed only if it does not introduce contradictions or change the meaning.
2) 'partially correct': if the answer is somewhat related but incomplete or inaccurate
3) 'incorrect': if the answer is wrong or unrelated
Do not include any additional text, explanation, or formatting. Only respond with one of the three words above."""
)


def submitted_answer(state: TaskState) -> str | None:
    """Return the last answer sent with the `submit` tool, or None if there is none."""
    for message in reversed(state.messages):
        if (
            isinstance(message, ChatMessageTool)
            and message.function == SUBMIT_TOOL
            and message.error is None
        ):
            return message.text
    return None


def parse_judge_verdict(response: str) -> bool:
    """Read the judge's verdict the way the upstream harness does.

    Only "correct" passes. "incorrect" and "partially correct" both contain
    "correct", so negative verdicts are checked first when falling back to
    substring matching.
    """
    verdict = response.strip().lower()
    if verdict == "correct":
        return True
    if verdict in ("partially correct", "incorrect"):
        return False
    if "incorrect" in verdict or "partially correct" in verdict:
        return False
    return "correct" in verdict


async def llm_judge(
    grader: Model, answer: str, references: Mapping[str, Any], question: str
) -> bool:
    """Grade `answer` against each `fuzzy_match` reference; any match passes."""
    golds = references.get("fuzzy_match")
    if golds is None:
        raise ValueError("llm_judge requires 'fuzzy_match' in reference_answers")
    if isinstance(golds, str):
        golds = [golds]

    for gold in golds:
        if gold == "N/A":
            if answer.strip().upper() == "N/A":
                return True
            continue
        output = await grader.generate(
            [
                ChatMessageSystem(content="You are a helpful assistant"),
                ChatMessageUser(
                    content=JUDGE_PROMPT.format(
                        question=question, reference=gold, answer=answer
                    )
                ),
            ],
            config=GenerateConfig(temperature=0.0, max_tokens=768),
        )
        if parse_judge_verdict(output.completion):
            return True
    return False


@scorer(
    metrics=[
        accuracy(),
        stderr(),
        grouped(accuracy(), "ui_version", all=False, name_template="v{group_name}"),
        grouped(accuracy(), "category", all=False),
    ]
)
def timewarp_scorer() -> Scorer:
    """Score the answer the agent sent with the `submit` tool.

    Each sample lists its verifiers in `metadata["eval"]["eval_types"]`; the
    answer must pass all of them. `string_match`, `number_match` and
    `list_match` are deterministic. `llm_judge` (2 of 231 goals) asks the
    `grader` model role, which defaults to the paper's GPT-5.1 judge.

    As in the upstream harness, only a message sent to the user counts as an
    answer: a sample without a `submit` tool call scores 0.
    """

    async def score(state: TaskState, target: Target) -> Score:
        answer = submitted_answer(state)
        if answer is None or not answer.strip():
            return Score(
                value=INCORRECT,
                answer=answer,
                explanation="No answer was sent with the submit tool.",
            )

        spec = state.metadata["eval"]
        eval_types: list[str] = spec["eval_types"]
        if not eval_types:
            raise ValueError("eval_types must list at least one verifier")
        references = spec["reference_answers"]

        results: dict[str, bool] = {}
        for eval_type in eval_types:
            if eval_type in DETERMINISTIC_VERIFIERS:
                results[eval_type] = DETERMINISTIC_VERIFIERS[eval_type](
                    answer, references
                )
            elif eval_type == "llm_judge":
                results[eval_type] = await llm_judge(
                    get_model(role="grader", default=DEFAULT_GRADER_MODEL),
                    answer,
                    references,
                    state.metadata["intent"],
                )
            else:
                raise ValueError(f"Unsupported TimeWarp eval_type {eval_type!r}")

        passed = all(results.values())
        return Score(
            value=CORRECT if passed else INCORRECT,
            answer=answer,
            explanation=", ".join(
                f"{name}: {'pass' if ok else 'fail'}" for name, ok in results.items()
            ),
            metadata={"verifiers": results},
        )

    return score
