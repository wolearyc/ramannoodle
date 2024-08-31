"""Exceptions, warnings, and related functions."""

from typing import Any, Sequence

from numpy.typing import NDArray


class NoMatchingLineFoundException(Exception):
    """No matching line can be found in file."""


class InvalidFileException(Exception):
    """File cannot be read, likely due to due to invalid or unexpected format."""


class InvalidDOFException(Exception):
    """A supplied degree of freedom is invalid."""


class DOFWarning(UserWarning):
    """A degree of freedom may not have been specified as intended."""


class SymmetryException(Exception):
    """Symmetry operation failed."""


class UsageError(Exception):
    """The user has done something they shouldn't.

    This exception is used sparingly, as (ideally) the structure of the API should
    dictate what the user should and shouldn't do.

    """


def _shape_string(shape: Sequence[int | None]) -> str:
    """Get a string representing a shape.

    Maps None --> "_", indicating that this element can
    be anything.
    """
    result = "("
    for i in shape:
        if i is None:
            result += "_,"
        else:
            result += f"{i},"
    if len(shape) == 1:
        return result + ")"
    return result[:-1] + ")"


def get_type_error(name: str, value: Any, correct_type: str) -> TypeError:
    """Return TypeError for an ndarray argument.

    :meta private:
    """
    wrong_type = type(value).__name__
    return TypeError(f"{name} should have type {correct_type}, not {wrong_type}")


def get_shape_error(name: str, array: NDArray, desired_shape: str) -> ValueError:
    """Return ValueError for an ndarray with the wrong shape.

    :meta private:
    """
    shape_spec = f"{_shape_string(array.shape)} != {desired_shape}"
    return ValueError(f"{name} has wrong shape: {shape_spec}")


def verify_ndarray(name: str, array: NDArray) -> None:
    """Verify type of NDArray .

    :meta private: We should avoid calling this function wherever possible (EATF)
    """
    try:
        _ = array.shape
    except AttributeError as exc:
        raise get_type_error(name, array, "ndarray") from exc


def verify_ndarray_shape(
    name: str, array: NDArray, shape: Sequence[int | None]
) -> None:
    """Verify an NDArray's shape.

    :meta private: We should avoid calling this function whenever possible (EATF).

    Parameters
    ----------
    shape
        int elements will be checked, None elements will not be.
    """
    try:
        if len(shape) != array.ndim:
            raise get_shape_error(name, array, _shape_string(shape))
        for d1, d2 in zip(array.shape, shape, strict=True):
            if d2 is not None and d1 != d2:
                raise get_shape_error(name, array, _shape_string(shape))
    except AttributeError as exc:
        raise get_type_error(name, array, "ndarray") from exc


def verify_list_len(name: str, array: list[Any], length: int | None) -> None:
    """Verify an list's shape.

    We should avoid calling this function whenever possible (EATF).

    :meta private:

    Parameters
    ----------
    length
        length. If None, not checked.
    """
    if not isinstance(array, list):
        raise get_type_error(name, array, "list")
    if length is not None and len(array) is not length:
        raise ValueError(f"{name} has wrong length: {len(array)} != length")


def verify_positions(name: str, array: NDArray) -> None:
    """Verify fractional positions according to dimensions and boundary conditions.

    :meta private:

    """
    verify_ndarray_shape(name, array, (None, 3))
    if (0 > array).any() or (array > 1.0).any():
        raise ValueError(f"{name} has coordinates that are not between 0 and 1")
