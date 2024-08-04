"""Fixtures for loading state."""

from pathlib import Path
from typing import TextIO
from collections.abc import Generator

import pytest
from pytest import FixtureRequest

from ramannoodle.symmetry import StructuralSymmetry
from ramannoodle import io


@pytest.fixture(scope="session")
def outcar_path_fixture(request: FixtureRequest) -> Path:
    """Return an outcar path."""
    return Path(request.param)


@pytest.fixture(scope="session")
def outcar_file_fixture(
    request: FixtureRequest,
) -> Generator[TextIO, None, None]:
    """Return an outcar file."""
    file = open(  # pylint: disable=consider-using-with
        Path(request.param), "r", encoding="utf-8"
    )
    yield file
    file.close()


# HACK: indirect fixtures are unable to be scoped, so manually cache.
symmetry_cache = {}


@pytest.fixture(scope="session")
def outcar_symmetry_fixture(request: FixtureRequest) -> StructuralSymmetry:
    """Return a structural symmetry."""
    if request.param not in symmetry_cache:
        symmetry_cache[request.param] = io.read_structural_symmetry(
            request.param, file_format="outcar"
        )
    return symmetry_cache[request.param]
