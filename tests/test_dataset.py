from typing import Any

import pytest

from timewarp.dataset import (
    LAST_TEST_TASK_ID,
    Split,
    load_records,
    record_to_samples,
    resolve_start_url,
    task_category,
    timewarp_dataset,
)
from timewarp.verifiers import DETERMINISTIC_VERIFIERS

# Task 1 of `data.json` at the pinned dataset revision (annotation notes trimmed).
TASK_1: dict[str, Any] = {
    "task_id": 1,
    "intent": "Is biophysics mentioned as a related page or branch in the Biology article, the Physics article, both, or neither of them?",
    "intent_template_id": 1,
    "sites": ["wiki"],
    "start_url": "__WIKI__",
    "eval": {
        "eval_types": ["string_match"],
        "reference_answers": {
            "must_include": ["biology"],
            "must_exclude": [
                "both",
                "neither",
                "^[\\s\\S]*\\bnot\\b (in |mentioned in |listed in |on |under )?(the )?biology\\b[\\s\\S]*$",
            ],
            "scope": "first_sentence",
            "fuzzy_match": "Biology",
        },
        "revision": 1,
        "annotation": {"confidence": "medium", "source": "batch_14.json"},
    },
}


def test_record_to_samples_creates_one_sample_per_version() -> None:
    samples = record_to_samples(TASK_1, ui_versions=[1, 6])

    assert [sample.id for sample in samples] == ["1_v1", "1_v6"]
    sample = samples[0]
    assert isinstance(sample.input, str)
    assert sample.input.startswith(TASK_1["intent"])
    assert "WIKI URL: http://localhost:5000" in sample.input
    assert "SHOP URL: http://localhost:5002/abc" in sample.input
    assert sample.input.endswith("Start URL: http://localhost:5000")
    assert sample.target == "Biology"
    assert sample.metadata is not None
    assert sample.metadata["ui_version"] == 1
    assert samples[1].metadata is not None
    assert samples[1].metadata["ui_version"] == 6
    assert sample.metadata["split"] == "test"
    assert sample.metadata["category"] == "wiki"
    assert sample.metadata["eval"] == TASK_1["eval"]


def test_plans_are_not_shown_to_the_agent() -> None:
    record = {
        **TASK_1,
        "task_id": 104,
        "additional_instructions": "Perform these instructions: send 'Biology'.",
    }
    [sample] = record_to_samples(record, ui_versions=[1])
    assert isinstance(sample.input, str)
    assert "Perform these instructions" not in sample.input
    assert sample.metadata is not None
    assert sample.metadata["split"] == "train"


@pytest.mark.parametrize(
    ("sites", "category"),
    [
        (["wiki"], "wiki"),
        (["news"], "news"),
        (["webshop"], "shop"),
        (["wiki", "news"], "multi"),
        (["news", "webshop"], "multi"),
    ],
)
def test_task_category(sites: list[str], category: str) -> None:
    assert task_category(sites) == category


def test_resolve_start_url() -> None:
    assert resolve_start_url("__NEWS__") == "http://localhost:5001"
    assert resolve_start_url("__WEBSHOP__") == "http://localhost:5002/abc"


@pytest.mark.dataset_download
def test_pinned_dataset_contents() -> None:
    records = load_records()

    assert len(records) == 231
    task_ids = sorted(record["task_id"] for record in records)
    assert task_ids == list(range(1, 232))
    # Plans (which contain answers) exist only for train goals.
    assert all(
        "additional_instructions" not in record
        for record in records
        if record["task_id"] <= LAST_TEST_TASK_ID
    )
    for record in records:
        spec = record["eval"]
        assert spec["eval_types"], record["task_id"]
        for eval_type in spec["eval_types"]:
            assert eval_type in {*DETERMINISTIC_VERIFIERS, "llm_judge"}, record[
                "task_id"
            ]
        assert isinstance(spec["reference_answers"]["fuzzy_match"], str)


@pytest.mark.dataset_download
@pytest.mark.parametrize(
    ("split", "expected"),
    [("test", 103 * 6), ("train", 128 * 6), ("all", 231 * 6)],
)
def test_split_sizes(split: Split, expected: int) -> None:
    dataset = timewarp_dataset(
        split,
        ui_versions=[1, 2, 3, 4, 5, 6],
        categories=["wiki", "news", "shop", "multi"],
    )
    assert len(dataset) == expected


@pytest.mark.dataset_download
def test_test_split_category_sizes() -> None:
    """Category sizes match Figure 2 of the paper (test split, all versions)."""
    expected = {"wiki": 186, "news": 132, "shop": 162, "multi": 138}
    for category, size in expected.items():
        dataset = timewarp_dataset("test", [1, 2, 3, 4, 5, 6], [category])
        assert len(dataset) == size, category
