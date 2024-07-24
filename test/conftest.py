"""Fixtures for loading state."""

from pathlib import Path
from typing import TextIO
from collections.abc import Generator

import pytest


@pytest.fixture
def phonons_outcar_path_fixture() -> Path:
    """Returns an outcar path"""
    return Path("test/data/TiO2_OUTCAR")


@pytest.fixture
def eps_outcar_path_fixture() -> Path:
    """Returns an outcar path"""
    return Path("test/data/EPS_OUTCAR")


@pytest.fixture(scope="function")
def phonons_outcar_file_fixture(
    phonons_outcar_path_fixture: Path,  # pylint: disable = redefined-outer-name
) -> Generator[TextIO, None, None]:
    """Returns an outcar file"""
    file = open(  # pylint: disable=consider-using-with
        phonons_outcar_path_fixture, "r", encoding="utf-8"
    )
    yield file
    file.close()


@pytest.fixture(scope="function")
def eps_outcar_file_fixture(
    eps_outcar_path_fixture: Path,  # pylint: disable = redefined-outer-name
) -> Generator[TextIO, None, None]:
    """Returns an outcar file"""
    file = open(  # pylint: disable=consider-using-with
        eps_outcar_path_fixture, "r", encoding="utf-8"
    )
    yield file
    file.close()
