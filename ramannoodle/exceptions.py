"""Exceptions and warnings for ramannoodle"""


class NoMatchingLineFoundException(Exception):
    """Raised when no line can be found in file."""

    def __init__(self, pattern: str):
        pass
