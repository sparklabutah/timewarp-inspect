from typing import Any

import pytest
from inspect_ai import eval
from inspect_ai.model import ChatMessageTool, ModelOutput, get_model
from inspect_ai.scorer import CORRECT

from timewarp import timewarp
from timewarp.agent import OBSERVATION_HEADER, OBSERVATION_METADATA_KEY


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"split": "dev"}, "split must be"),
        ({"ui_versions": 7}, "ui_versions must be"),
        ({"ui_versions": []}, "ui_versions must be"),
        ({"categories": "forum"}, "categories must be"),
        ({"max_steps": 0}, "max_steps must be"),
    ],
)
def test_invalid_parameters(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        timewarp(**kwargs)


@pytest.mark.dataset_download
def test_task_selection() -> None:
    task = timewarp(ui_versions=[2, 5], categories="wiki")
    assert len(task.dataset) == 31 * 2
    assert {
        sample.metadata["ui_version"] for sample in task.dataset if sample.metadata
    } == {2, 5}
    assert task.turn_limit == 150


@pytest.mark.docker
@pytest.mark.slow
@pytest.mark.dataset_download
def test_end_to_end_with_scripted_agent() -> None:
    """Run the sandbox, act on the Wiki, and submit the gold answer for task 1."""
    model = get_model(
        "mockllm/model",
        custom_outputs=[
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="browser_navigate",
                tool_arguments={"url": "http://localhost:5000/random"},
            ),
            ModelOutput.for_tool_call(
                model="mockllm/model",
                tool_name="submit",
                tool_arguments={"answer": "Biology."},
            ),
        ],
    )
    [log] = eval(
        timewarp(ui_versions=3, categories="wiki"),
        model=model,
        sample_id="1_v3",
    )

    assert log.status == "success", log.error
    assert log.samples is not None
    [sample] = log.samples
    navigate_results = [
        message
        for message in sample.messages
        if isinstance(message, ChatMessageTool)
        and message.function == "browser_navigate"
    ]
    assert len(navigate_results) == 1
    assert navigate_results[0].error is None
    observations = [
        message
        for message in sample.messages
        if message.metadata and message.metadata.get(OBSERVATION_METADATA_KEY)
    ]
    assert len(observations) == 2
    assert observations[0].text.startswith(OBSERVATION_HEADER)
    assert "Page URL: http://localhost:5000" in observations[0].text
    assert "Page URL: http://localhost:5000/random" in observations[1].text or (
        "Page URL: http://localhost:5000/wiki/" in observations[1].text
    )
    assert sample.scores is not None
    assert sample.scores["timewarp_scorer"].value == CORRECT
