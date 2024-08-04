"""Defines some useful globals."""

from typing import Sequence

from numpy.typing import NDArray

ATOMIC_WEIGHTS = {
    "H": 1.008,
    "He": 4.002602,
    "Li": 6.94,
    "Be": 9.0121831,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998403163,
    "Ne": 20.1797,
    "Na": 22.98976928,
    "Mg": 24.305,
    "Al": 26.9815385,
    "Si": 28.085,
    "P": 30.973761998,
    "S": 32.06,
    "Cl": 35.45,
    "Ar": 39.948,
    "K": 39.0983,
    "Ca": 40.078,
    "Sc": 44.955908,
    "Ti": 47.867,
    "V": 50.9415,
    "Cr": 51.9961,
    "Mn": 54.938044,
    "Fe": 55.845,
    "Co": 58.933194,
    "Ni": 58.6934,
    "Cu": 63.546,
    "Zn": 65.38,
    "Ga": 69.723,
    "Ge": 72.63,
    "As": 74.921595,
    "Se": 78.971,
    "Br": 79.904,
    "Kr": 83.798,
    "Rb": 85.4678,
    "Sr": 87.62,
    "Y": 88.90584,
    "Zr": 91.224,
    "Nb": 92.90637,
    "Mo": 95.95,
    "Tc": 97.90721,
    "Ru": 101.07,
    "Rh": 102.9055,
    "Pd": 106.42,
    "Ag": 107.8682,
    "Cd": 112.414,
    "In": 114.818,
    "Sn": 118.71,
    "Sb": 121.76,
    "Te": 127.6,
    "I": 126.90447,
    "Xe": 131.293,
    "Cs": 132.90545196,
    "Ba": 137.327,
    "La": 138.90547,
    "Ce": 140.116,
    "Pr": 140.90766,
    "Nd": 144.242,
    "Pm": 144.91276,
    "Sm": 150.36,
    "Eu": 151.964,
    "Gd": 157.25,
    "Tb": 158.92535,
    "Dy": 162.5,
    "Ho": 164.93033,
    "Er": 167.259,
    "Tm": 168.93422,
    "Yb": 173.054,
    "Lu": 174.9668,
    "Hf": 178.49,
    "Ta": 180.94788,
    "W": 183.84,
    "Re": 186.207,
    "Os": 190.23,
    "Ir": 192.217,
    "Pt": 195.084,
    "Au": 196.966569,
    "Hg": 200.592,
    "Tl": 204.38,
    "Pb": 207.2,
    "Bi": 208.9804,
    "Po": 208.98243,
    "At": 209.98715,
    "Rn": 222.01758,
    "Fr": 223.01974,
    "Ra": 226.02541,
    "Ac": 227.02775,
    "Th": 232.0377,
    "Pa": 231.03588,
    "U": 238.02891,
    "Np": 237.04817,
    "Pu": 244.06421,
    "Am": 243.06138,
    "Cm": 247.07035,
    "Bk": 247.07031,
    "Cf": 251.07959,
    "Es": 252.083,
    "Fm": 257.09511,
    "Md": 258.09843,
    "No": 259.101,
    "Lr": 262.11,
    "Rf": 267.122,
    "Db": 268.126,
    "Sg": 271.134,
    "Bh": 270.133,
    "Hs": 269.1338,
    "Mt": 278.156,
    "Ds": 281.165,
    "Rg": 281.166,
    "Cn": 285.177,
    "Nh": 286.182,
    "Fl": 289.19,
    "Mc": 289.194,
    "Lv": 293.204,
    "Ts": 293.208,
    "Og": 294.214,
}

ATOMIC_NUMBERS = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cn": 112,
    "Nh": 113,
    "Fl": 114,
    "Mc": 115,
    "Lv": 116,
    "Ts": 117,
    "Og": 118,
}

RAMAN_TENSOR_CENTRAL_DIFFERENCE = 0.001
BOLTZMANN_CONSTANT = 8.617333262e-5  # Units: eV/K


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
    return result[:-1] + ")"


def verify_ndarray(name: str, array: NDArray) -> None:
    """Verify type of NDArray .

    We should avoid calling this function wherever possible (EATF)
    """
    try:
        _ = array.shape
    except AttributeError as exc:
        wrong_type = type(array).__name__
        raise TypeError(f"{name} should be an ndarray, not a {wrong_type}") from exc


def verify_ndarray_shape(
    name: str, array: NDArray, shape: Sequence[int | None]
) -> None:
    """Verify an NDArray's shape.

    We should avoid calling this function whenever possible (EATF).

    Parameters
    ----------
    shape
        int elements will be checked, None elements will not be.
    """
    try:
        if len(shape) != array.ndim:
            shape_spec = f"{_shape_string(array.shape)} != {_shape_string(shape)}"
            raise ValueError(f"{name} has wrong shape: {shape_spec}")
        for d1, d2 in zip(array.shape, shape, strict=True):
            if d2 is not None and d1 != d2:
                shape_spec = f"{_shape_string(array.shape)} != {_shape_string(shape)}"
                raise ValueError(f"{name} has wrong shape: {shape_spec}")
    except AttributeError as exc:
        wrong_type = type(array).__name__
        raise TypeError(f"{name} should be an ndarray, not a {wrong_type}") from exc
