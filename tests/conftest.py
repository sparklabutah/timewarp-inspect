import os

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--docker",
        action="store_true",
        default=False,
        help="Run tests that build and start the TimeWarp Docker sandbox.",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if config.getoption("--docker") or os.environ.get("RUN_DOCKER_TESTS") == "1":
        return
    skip_docker = pytest.mark.skip(reason="needs --docker or RUN_DOCKER_TESTS=1")
    for item in items:
        if "docker" in item.keywords:
            item.add_marker(skip_docker)
