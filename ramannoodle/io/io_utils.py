"""Universal IO utility functions."""

from typing import TextIO
from pathlib import Path

from ramannoodle.exceptions import NoMatchingLineFoundException


def _skip_file_until_line_contains(file: TextIO, content: str) -> str:
    """Read through a file until a line containing specific content is found."""
    for line in file:
        if content in line:
            return line
    raise NoMatchingLineFoundException(content)


def pathify(filepath: str | Path) -> Path:
    """Convert filepath to Path.

    :meta private:
    """
    return Path(filepath)


def pathify_as_list(filepaths: str | Path | list[str] | list[Path]) -> list[Path]:
    """Convert filepaths to list of Paths.

    :meta private:
    """
    if isinstance(filepaths, list):
        paths = []
        for item in filepaths:
            try:
                paths.append(Path(item))
            except TypeError as exc:
                raise TypeError(f"{item} cannot be resolved as a filepath") from exc
        return paths
    try:
        return [Path(filepaths)]
    except TypeError as exc:
        raise TypeError(f"{filepaths} cannot be resolved as a filepath") from exc
