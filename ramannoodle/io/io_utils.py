"""Universal IO utility functions."""

from typing import TextIO
from ..exceptions import NoMatchingLineFoundException


def _skip_file_until_line_contains(file: TextIO, content: str) -> str:
    """Read through a file until a line containing specific content is found."""
    for line in file:
        if content in line:
            return line
    raise NoMatchingLineFoundException(content)
