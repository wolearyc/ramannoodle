"""Fixtures for loading state."""

from pathlib import Path
from typing import TextIO
from collections.abc import Generator

import pytest
from pytest import FixtureRequest


@pytest.fixture
def outcar_path_fixture(request: FixtureRequest) -> Path:
    """Return an outcar path."""
    return Path(request.param)


@pytest.fixture
def outcar_file_fixture(
    request: FixtureRequest,  # pylint: disable = redefined-outer-name
) -> Generator[TextIO, None, None]:
    """Return an outcar file."""
    file = open(  # pylint: disable=consider-using-with
        Path(request.param), "r", encoding="utf-8"
    )
    yield file
    file.close()
