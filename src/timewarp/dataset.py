"""TimeWarp dataset: 231 goals, each rendered in six UI versions."""

import json
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from inspect_ai.dataset import Dataset, MemoryDataset, Sample

DATASET_REPO = "sparklabutah/timewarp"
DATASET_REVISION = "246edb1cc9c4746df68172dad661c97164064cec"
DATASET_FILE = "data.json"

UI_VERSIONS = (1, 2, 3, 4, 5, 6)
CATEGORIES = ("wiki", "news", "shop", "multi")

# Goals 1-103 form the test split and 104-231 the train split, as in the
# BrowserGym `timewarp.csv` benchmark metadata. Train goals carry
# `additional_instructions` (human-refined plans that include the answer) for
# collecting teacher trajectories; they are never shown to the evaluated agent.
LAST_TEST_TASK_ID = 103

# Ports used by the upstream `scripts/environment/run_all_env.sh`.
SITE_URLS = {
    "wiki": "http://localhost:5000",
    "news": "http://localhost:5001",
    "webshop": "http://localhost:5002/abc",
}

Split = Literal["test", "train", "all"]

TASK_PROMPT = """{intent}

The task takes place on the following websites. Only navigate to URLs on these websites; do not navigate to external websites.
WIKI URL: {wiki_url}
NEWS URL: {news_url}
SHOP URL: {shop_url}

Start URL: {start_url}"""


def load_records() -> list[dict[str, Any]]:
    """Download the pinned TimeWarp task file from Hugging Face."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=DATASET_REPO,
        filename=DATASET_FILE,
        repo_type="dataset",
        revision=DATASET_REVISION,
    )
    with open(path, encoding="utf-8") as f:
        records: list[dict[str, Any]] = json.load(f)
    return records


def task_split(task_id: int) -> Literal["test", "train"]:
    return "test" if task_id <= LAST_TEST_TASK_ID else "train"


def task_category(sites: Sequence[str]) -> str:
    if len(sites) > 1:
        return "multi"
    return "shop" if sites[0] == "webshop" else sites[0]


def resolve_start_url(start_url: str) -> str:
    """Replace the `__WIKI__`/`__NEWS__`/`__WEBSHOP__` placeholders with local URLs."""
    for site, url in SITE_URLS.items():
        start_url = start_url.replace(f"__{site.upper()}__", url)
    return start_url


def record_to_samples(
    record: dict[str, Any], ui_versions: Iterable[int]
) -> list[Sample]:
    """Turn one TimeWarp goal into one sample per requested UI version."""
    task_id = int(record["task_id"])
    start_url = resolve_start_url(record["start_url"])
    prompt = TASK_PROMPT.format(
        intent=record["intent"],
        wiki_url=SITE_URLS["wiki"],
        news_url=SITE_URLS["news"],
        shop_url=SITE_URLS["webshop"],
        start_url=start_url,
    )
    metadata = {
        "task_id": task_id,
        "split": task_split(task_id),
        "category": task_category(record["sites"]),
        "sites": record["sites"],
        "start_url": start_url,
        "intent": record["intent"],
        "eval": record["eval"],
    }
    return [
        Sample(
            id=f"{task_id}_v{ui_version}",
            input=prompt,
            target=record["eval"]["reference_answers"]["fuzzy_match"],
            metadata={**metadata, "ui_version": ui_version},
        )
        for ui_version in ui_versions
    ]


def timewarp_dataset(
    split: Split,
    ui_versions: Sequence[int],
    categories: Sequence[str],
) -> Dataset:
    """Build the TimeWarp dataset for the given split, UI versions and categories."""
    samples = [
        sample
        for record in load_records()
        if split in ("all", task_split(int(record["task_id"])))
        and task_category(record["sites"]) in categories
        for sample in record_to_samples(record, ui_versions)
    ]
    return MemoryDataset(samples=samples, name="timewarp")
