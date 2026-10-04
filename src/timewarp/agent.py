"""Default TimeWarp agent: one browser action per step over Playwright MCP tools.

The loop follows the BrowserGym/AgentLab protocol used in the paper:

- The start page is open before the first step, and the agent sees it.
- Each step executes at most one action: only the first tool call of a turn runs.
- A turn without a usable tool call is retried within the same step, up to
  `max_retries` times (AgentLab's GenericAgent retries unparseable responses
  up to 4 times). The step is used up once the retries run out.
- After every action the agent receives the accessibility snapshot of the
  current page in a separate message.
- Only the latest page observation is sent to the model. Earlier observations
  are replaced with a short note, while the agent's own messages and the
  results of its actions are kept.
"""

from collections.abc import Callable, Sequence

from inspect_ai.model import (
    ChatMessage,
    ChatMessageSystem,
    ChatMessageTool,
    ChatMessageUser,
    Content,
    ContentImage,
    ContentText,
    Model,
    execute_tools,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver
from inspect_ai.tool import (
    Tool,
    ToolCall,
    ToolCallError,
    ToolDef,
    ToolError,
    ToolResult,
    ToolSource,
    mcp_connection,
    mcp_server_sandbox,
    mcp_tools,
    tool,
)

from timewarp.dataset import SITE_URLS

# Browser tools offered to the agent. They cover the BrowserGym action space
# used in the paper (goto, go_back, click, fill, press, tab management); the
# page is observed automatically after each action. Tools that run page
# JavaScript, read network traffic or upload files are left out.
BROWSER_TOOLS = [
    "browser_navigate",
    "browser_navigate_back",
    "browser_click",
    "browser_type",
    "browser_select_option",
    "browser_hover",
    "browser_press_key",
    "browser_handle_dialog",
    "browser_wait_for",
    "browser_tabs",
]
SCREENSHOT_TOOLS = ["browser_take_screenshot"]
SNAPSHOT_TOOL = "browser_snapshot"

# Retries of a turn without a usable tool call, as in AgentLab's GenericAgent.
DEFAULT_MAX_RETRIES = 4

# The page themes load stylesheets, scripts and fonts from public CDNs, and
# WebShop product images come from Amazon's image hosts, as in the upstream
# environments. All other origins are blocked, which keeps the agent on the
# TimeWarp sites (the upstream task scores 0 when a tab leaves them).
ASSET_ORIGINS = [
    "https://ajax.googleapis.com",
    "https://cdnjs.cloudflare.com",
    "https://fonts.googleapis.com",
    "https://fonts.gstatic.com",
    "https://images-na.ssl-images-amazon.com",
    "https://images.unsplash.com",
    "https://m.media-amazon.com",
    "https://maxcdn.bootstrapcdn.com",
    "https://stackpath.bootstrapcdn.com",
]

AGENT_INSTRUCTIONS = """You are a web agent. You complete tasks for a user by operating a web browser with the browser tools.

After each action you receive the accessibility snapshot of the current page. Refer to page elements by the `ref` values shown in the snapshot. Only the latest snapshot is shown; earlier ones are omitted, so note anything you need to remember in your reasoning.

Before each action, briefly reason about the current page, your plan, and anything you need to remember. Perform exactly one action per response. Strictly follow the instructions in the task description.

When you have completed the task, send your final message to the user with the `submit()` tool. The message is evaluated, so it must contain the answer the task asks for."""

OBSERVATION_HEADER = "# Current page"
OMITTED_OBSERVATION = (
    "(Page observation omitted. Only the latest observation is shown.)"
)
OMITTED_SCREENSHOT = "(Screenshot omitted. Only the latest screenshot is shown.)"
NO_ACTION_PROMPT = "No action was taken. Use one of the browser tools, or send your final message with the `submit()` tool."
EXTRA_CALL_ERROR = "Only one action is allowed per step, so this call was not executed."

# Marks the messages that carry a page observation, so that pruning never
# depends on matching their text.
OBSERVATION_METADATA_KEY = "timewarp_observation"


def _site_origins() -> list[str]:
    return [url.removesuffix("/abc") for url in SITE_URLS.values()]


def browser_tools(screenshots: bool) -> ToolSource:
    """Playwright MCP tools, served from the sample's `default` sandbox.

    Includes `browser_snapshot`, which the agent loop uses to observe the page.

    Args:
        screenshots: Also provide `browser_take_screenshot`, for vision-capable models.
    """
    server = mcp_server_sandbox(
        name="playwright",
        command="node",
        args=[
            "/app/cli.js",
            "--headless",
            "--browser=chromium",
            "--no-sandbox",
            "--isolated",
            "--viewport-size=1280x720",
            "--allowed-origins=" + ";".join(_site_origins() + ASSET_ORIGINS),
            # The agent loop takes an explicit snapshot after each action.
            "--snapshot-mode=none",
            "--codegen=none",
            "--image-responses=" + ("allow" if screenshots else "omit"),
            "--output-dir=/tmp/playwright-mcp",
        ],
        cwd="/home/node",
    )
    names = BROWSER_TOOLS + [SNAPSHOT_TOOL] + (SCREENSHOT_TOOLS if screenshots else [])
    return mcp_tools(server, tools=names)


@tool(name="submit")
def submit() -> Tool:
    async def execute(answer: str) -> str:
        """Send your final message to the user. The message is evaluated.

        Args:
            answer: Your final message to the user, containing the answer the task asks for.
        """
        return answer

    return execute


@solver
def browser_agent(
    max_steps: int = 30,
    max_retries: int = DEFAULT_MAX_RETRIES,
    screenshots: bool = False,
) -> Solver:
    """Browse the TimeWarp sites one action per step and submit a free-text answer.

    The answer is sent with the `submit` tool, where `timewarp_scorer()` reads it.

    Args:
        max_steps: Maximum number of steps, including the final answer.
        max_retries: Retries within a step when a turn has no usable tool call.
        screenshots: Also provide `browser_take_screenshot`, for vision-capable models.
    """
    browser = browser_tools(screenshots)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        async with mcp_connection([browser]):
            return await run_browser_loop(
                state, get_model(), await browser.tools(), max_steps, max_retries
            )

    return solve


async def run_browser_loop(
    state: TaskState,
    model: Model,
    browser: Sequence[Tool],
    max_steps: int,
    max_retries: int,
) -> TaskState:
    """Run the agent loop with the given browser tools until it submits or runs out of steps.

    `browser` must include `browser_navigate` and `browser_snapshot`; the
    latter is used for observations and is not offered to the model.
    """
    snapshot = _find_tool(browser, SNAPSHOT_TOOL)
    await _find_tool(browser, "browser_navigate")(url=state.metadata["start_url"])
    state.tools = [t for t in browser if ToolDef(t).name != SNAPSHOT_TOOL] + [submit()]
    state.messages = [
        ChatMessageSystem(content=AGENT_INSTRUCTIONS),
        *state.messages,
        await _observe(snapshot),
    ]

    steps = 0
    retries = 0
    while steps < max_steps:
        output = await model.generate(
            input=latest_observation_only(state.messages), tools=state.tools
        )
        state.output = output
        state.messages.append(output.message)
        if output.stop_reason == "model_length":
            return state

        calls = output.message.tool_calls or []
        if not calls or calls[0].parse_error is not None:
            if calls:
                state.messages.extend(_rejected(calls))
            else:
                state.messages.append(ChatMessageUser(content=NO_ACTION_PROMPT))
            retries += 1
            if retries > max_retries:
                steps += 1
                retries = 0
            continue

        steps += 1
        retries = 0
        action = calls[0]
        executed = await execute_tools(
            [
                *state.messages[:-1],
                output.message.model_copy(update={"tool_calls": [action]}),
            ],
            state.tools,
        )
        state.messages.extend([*executed.messages, *_rejected(calls[1:])])

        result = executed.messages[-1]
        if action.function == "submit" and isinstance(result, ChatMessageTool):
            if result.error is None:
                state.output.completion = result.text
                return state
            continue

        state.messages.append(await _observe(snapshot))

    return state


def latest_observation_only(messages: Sequence[ChatMessage]) -> list[ChatMessage]:
    """Copy `messages`, keeping only the latest page observation and screenshot."""
    latest_observation = _last_index(messages, _is_observation)
    latest_screenshot = _last_index(messages, _has_image)
    pruned: list[ChatMessage] = []
    for i, message in enumerate(messages):
        if _is_observation(message) and i != latest_observation:
            pruned.append(message.model_copy(update={"content": OMITTED_OBSERVATION}))
        elif _has_image(message) and i != latest_screenshot:
            pruned.append(
                message.model_copy(update={"content": _without_images(message.content)})
            )
        else:
            pruned.append(message)
    return pruned


async def _observe(snapshot: Tool) -> ChatMessageUser:
    """Capture the current page as an observation message."""
    try:
        page = _result_text(await snapshot())
    except ToolError as error:
        page = f"The page snapshot could not be captured: {error.message}"
    return ChatMessageUser(
        content=f"{OBSERVATION_HEADER}\n{page}",
        metadata={OBSERVATION_METADATA_KEY: True},
    )


def _find_tool(tools: Sequence[Tool], name: str) -> Tool:
    for candidate in tools:
        if ToolDef(candidate).name == name:
            return candidate
    raise RuntimeError(f"Playwright MCP server does not provide the {name} tool")


def _rejected(calls: Sequence[ToolCall]) -> list[ChatMessageTool]:
    """Error results for tool calls that were not executed."""
    return [
        ChatMessageTool(
            content="",
            tool_call_id=call.id,
            function=call.function,
            error=ToolCallError("parsing", call.parse_error)
            if call.parse_error is not None
            else ToolCallError("unknown", EXTRA_CALL_ERROR),
        )
        for call in calls
    ]


def _result_text(result: ToolResult) -> str:
    if isinstance(result, str):
        return result
    items = result if isinstance(result, list) else [result]
    return "\n".join(item.text for item in items if isinstance(item, ContentText))


def _is_observation(message: ChatMessage) -> bool:
    return bool(message.metadata and message.metadata.get(OBSERVATION_METADATA_KEY))


def _has_image(message: ChatMessage) -> bool:
    return not isinstance(message.content, str) and any(
        isinstance(item, ContentImage) for item in message.content
    )


def _without_images(content: str | list[Content]) -> str | list[Content]:
    if isinstance(content, str):
        return content
    return [
        ContentText(text=OMITTED_SCREENSHOT) if isinstance(item, ContentImage) else item
        for item in content
    ]


def _last_index(
    messages: Sequence[ChatMessage], predicate: Callable[[ChatMessage], bool]
) -> int:
    return max((i for i, m in enumerate(messages) if predicate(m)), default=-1)
