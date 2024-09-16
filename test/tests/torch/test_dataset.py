"""Testing for PyTorch dataset."""

import pytest

import ramannoodle.io.generic as generic_io


@pytest.mark.parametrize(
    "filepaths, file_format",
    [
        (
            [
                "test/data/TiO2/O43_0.1x_eps_OUTCAR",
                "test/data/TiO2/O43_0.1y_eps_OUTCAR",
                "test/data/TiO2/O43_0.1z_eps_OUTCAR",
            ],
            "outcar",
        ),
        ("test/data/STO/vasprun.xml", "vasprun.xml"),
    ],
)
def test_load_polarizability_dataset(
    filepaths: str | list[str], file_format: str
) -> None:
    """Test of generic load_polarizability_dataset (normal)."""
    dataset = generic_io.read_polarizability_dataset(filepaths, file_format)
    if isinstance(filepaths, list):
        assert len(dataset) == len(filepaths)
    else:
        assert len(dataset) == 1
