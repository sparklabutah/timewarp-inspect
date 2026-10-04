"""Agent loop tests with fake browser tools (no Docker or MCP server needed)."""

from typing import Any

from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.log import EvalSample
from inspect_ai.model import (
    ChatCompletionChoice,
    ChatMessage,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    ContentImage,
    ContentText,
    ModelOutput,
    get_model,
)
from inspect_ai.scorer import CORRECT, INCORRECT
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import Tool, ToolCall, ToolError, tool

from timewarp import browser_agent
from timewarp.agent import (
    AGENT_INSTRUCTIONS,
    BROWSER_TOOLS,
    EXTRA_CALL_ERROR,
    NO_ACTION_PROMPT,
    OBSERVATION_HEADER,
    OBSERVATION_METADATA_KEY,
    OMITTED_OBSERVATION,
    OMITTED_SCREENSHOT,
    latest_observation_only,
    run_browser_loop,
)
from timewarp.scorer import timewarp_scorer

MODEL = "mockllm/model"


class FakeBrowser:
    """Records actions and serves a page whose content is the current URL."""

    def __init__(self, snapshot_error: bool = False) -> None:
        self.url = "about:blank"
        self.clicks: list[str] = []
        self.snapshot_error = snapshot_error

    def tools(self) -> list[Tool]:
        @tool(name="browser_navigate")
        def navigate() -> Tool:
            async def execute(url: str) -> str:
                """Navigate to a URL.

                Args:
                    url: URL to open.
                """
                self.url = url
                return f"Navigated to {url}"

            return execute

        @tool(name="browser_snapshot")
        def snapshot() -> Tool:
            async def execute() -> str:
                """Capture the accessibility snapshot of the current page."""
                if self.snapshot_error and self.url != "http://localhost:5000":
                    raise ToolError("Target page has been closed")
                return f"- Page URL: {self.url}"

            return execute

        @tool(name="browser_click")
        def click() -> Tool:
            async def execute(target: str) -> str:
                """Click an element.

                Args:
                    target: Element reference.
                """
                if target == "missing":
                    raise ToolError(
                        f"Ref {target} not found in the current page snapshot"
                    )
                self.clicks.append(target)
                self.url = f"{self.url}/{target}"
                return f"Clicked {target}"

            return execute

        return [navigate(), snapshot(), click()]


@solver
def fake_browser_agent(
    browser: FakeBrowser, max_steps: int, max_retries: int
) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        return await run_browser_loop(
            state, get_model(), browser.tools(), max_steps, max_retries
        )

    return solve


def tool_turn(*calls: tuple[str, dict[str, Any]]) -> ModelOutput:
    message = ChatMessageAssistant(
        content="",
        tool_calls=[
            ToolCall(id=f"call_{i}", function=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ],
    )
    return ModelOutput(
        model=MODEL,
        choices=[ChatCompletionChoice(message=message, stop_reason="tool_calls")],
    )


def text_turn(text: str = "Let me think.") -> ModelOutput:
    return ModelOutput.from_content(model=MODEL, content=text)


SUBMIT = tool_turn(("submit", {"answer": "Biology"}))


def run(
    outputs: list[ModelOutput],
    browser: FakeBrowser,
    max_steps: int = 5,
    max_retries: int = 4,
) -> EvalSample:
    spec = {
        "eval_types": ["string_match"],
        "reference_answers": {"must_include": ["biology"], "fuzzy_match": "Biology"},
    }
    task = Task(
        dataset=[
            Sample(
                id="1_v1",
                input="Which article mentions biophysics?",
                target="Biology",
                metadata={
                    "start_url": "http://localhost:5000",
                    "eval": spec,
                    "intent": "Which article mentions biophysics?",
                    "ui_version": 1,
                    "category": "wiki",
                },
            )
        ],
        solver=fake_browser_agent(browser, max_steps, max_retries),
        scorer=timewarp_scorer(),
    )
    [log] = eval(task, model=get_model(MODEL, custom_outputs=outputs))
    assert log.status == "success", log.error
    assert log.samples is not None
    [sample] = log.samples
    assert sample.error is None
    return sample


def score(sample: EvalSample) -> Any:
    assert sample.scores is not None
    return sample.scores["timewarp_scorer"].value


def observations(sample: EvalSample) -> list[ChatMessage]:
    return [
        m
        for m in sample.messages
        if m.metadata and m.metadata.get(OBSERVATION_METADATA_KEY)
    ]


def test_start_page_is_observed_before_the_first_step() -> None:
    sample = run([SUBMIT], FakeBrowser())

    assert isinstance(sample.messages[0], ChatMessageSystem)
    first_observation = sample.messages[2]
    assert isinstance(first_observation, ChatMessageUser)
    assert (
        first_observation.text
        == f"{OBSERVATION_HEADER}\n- Page URL: http://localhost:5000"
    )
    assert score(sample) == CORRECT


def test_only_the_first_tool_call_per_step_runs() -> None:
    browser = FakeBrowser()
    sample = run(
        [
            tool_turn(
                ("browser_click", {"target": "e1"}), ("browser_click", {"target": "e2"})
            ),
            SUBMIT,
        ],
        browser,
    )

    assert browser.clicks == ["e1"]
    results = [m for m in sample.messages if isinstance(m, ChatMessageTool)]
    assert results[0].error is None
    assert results[1].error is not None
    assert results[1].error.message == EXTRA_CALL_ERROR
    assert observations(sample)[-1].text.endswith("http://localhost:5000/e1")
    assert score(sample) == CORRECT


def test_failed_action_is_followed_by_an_observation() -> None:
    sample = run(
        [tool_turn(("browser_click", {"target": "missing"})), SUBMIT], FakeBrowser()
    )
    click_result = next(m for m in sample.messages if isinstance(m, ChatMessageTool))
    assert click_result.error is not None
    following = sample.messages[sample.messages.index(click_result) + 1]
    assert following.metadata is not None
    assert following.metadata[OBSERVATION_METADATA_KEY] is True


def test_snapshot_errors_become_observations() -> None:
    sample = run(
        [tool_turn(("browser_click", {"target": "e1"})), SUBMIT],
        FakeBrowser(snapshot_error=True),
    )
    assert "Target page has been closed" in observations(sample)[-1].text
    assert score(sample) == CORRECT


def test_turn_without_tool_call_is_retried_within_the_step() -> None:
    sample = run([text_turn(), SUBMIT], FakeBrowser(), max_steps=1)
    assert any(
        isinstance(m, ChatMessageUser) and m.text == NO_ACTION_PROMPT
        for m in sample.messages
    )
    assert score(sample) == CORRECT


def test_exhausted_retries_use_up_the_step() -> None:
    sample = run(
        [text_turn(), text_turn(), SUBMIT], FakeBrowser(), max_steps=1, max_retries=1
    )
    assert score(sample) == INCORRECT
    assert not any(
        isinstance(m, ChatMessageTool) and m.function == "submit"
        for m in sample.messages
    )


def test_step_limit_without_submission_scores_zero() -> None:
    browser = FakeBrowser()
    sample = run(
        [tool_turn(("browser_click", {"target": "e1"}))] * 6, browser, max_steps=3
    )
    assert len(browser.clicks) == 3
    assert score(sample) == INCORRECT


def test_snapshot_tool_is_not_offered_to_the_model() -> None:
    sample = run([tool_turn(("browser_snapshot", {})), SUBMIT], FakeBrowser())
    result = next(m for m in sample.messages if isinstance(m, ChatMessageTool))
    assert result.error is not None
    assert "browser_snapshot" not in BROWSER_TOOLS


def test_latest_observation_only_keeps_system_prompt_and_latest_page() -> None:
    def page(text: str) -> ChatMessageUser:
        return ChatMessageUser(
            content=f"{OBSERVATION_HEADER}\n{text}",
            metadata={OBSERVATION_METADATA_KEY: True},
        )

    messages: list[ChatMessage] = [
        ChatMessageSystem(
            content=AGENT_INSTRUCTIONS + f"\n{OBSERVATION_HEADER} mentioned"
        ),
        ChatMessageUser(content="task"),
        page("page one"),
        ChatMessageAssistant(content=f"I saw {OBSERVATION_HEADER} with a heading."),
        ChatMessageTool(
            content=[
                ContentImage(image="data:image/png;base64,AAAA"),
                ContentText(text="old"),
            ],
            tool_call_id="call_0",
            function="browser_take_screenshot",
        ),
        page("page two"),
        ChatMessageTool(
            content=[ContentImage(image="data:image/png;base64,BBBB")],
            tool_call_id="call_1",
            function="browser_take_screenshot",
        ),
        page("page three"),
    ]
    pruned = latest_observation_only(messages)

    assert pruned[0] is messages[0]
    assert pruned[1] is messages[1]
    assert pruned[2].text == OMITTED_OBSERVATION
    assert pruned[3] is messages[3]
    assert pruned[4].content == [
        ContentText(text=OMITTED_SCREENSHOT),
        ContentText(text="old"),
    ]
    assert pruned[5].text == OMITTED_OBSERVATION
    assert pruned[6] is messages[6]
    assert pruned[7] is messages[7]
    assert "page one" in messages[2].text


def test_browser_agent_builds() -> None:
    assert callable(browser_agent())
    assert callable(browser_agent(max_steps=10, max_retries=0, screenshots=True))


def test_browser_tools_exclude_page_scripting() -> None:
    assert "browser_evaluate" not in BROWSER_TOOLS
    assert "browser_run_code_unsafe" not in BROWSER_TOOLS
