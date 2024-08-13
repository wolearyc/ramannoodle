"""Fixtures for loading state."""

from pathlib import Path
from typing import TextIO
from collections.abc import Generator

import pytest
from pytest import FixtureRequest

from ramannoodle.symmetry.structural import ReferenceStructure
import ramannoodle.io.generic as rn_io


@pytest.fixture(scope="session")
def path_fixture(request: FixtureRequest) -> Path:
    """Return an outcar path."""
    return Path(request.param)


@pytest.fixture(scope="session")
def file_fixture(
    request: FixtureRequest,
) -> Generator[TextIO, None, None]:
    """Return a file."""
    file = open(  # pylint: disable=consider-using-with
        Path(request.param), "r", encoding="utf-8"
    )
    yield file
    file.close()


# HACK: indirect fixtures are unable to be scoped, so manually cache.
ref_structure_cache = {}


@pytest.fixture(scope="session")
def outcar_ref_structure_fixture(request: FixtureRequest) -> ReferenceStructure:
    """Return a reference structure."""
    if request.param not in ref_structure_cache:
        ref_structure_cache[request.param] = rn_io.read_ref_structure(
            request.param, file_format="outcar"
        )
    return ref_structure_cache[request.param]
