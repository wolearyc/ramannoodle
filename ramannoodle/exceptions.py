"""Exceptions and warnings for ramannoodle"""


class NoMatchingLineFoundException(Exception):
    """Raised when no line can be found in file."""

    def __init__(self, pattern: str):
        pass


class InvalidDOFException(Exception):
    """Raised when things a degree of freedom is invalid in some way."""

    def __init__(self, reason: str):
        pass
