"""Testing for GNN-based models."""

import os
from typing import Type
import re

import pytest

import numpy as np
from numpy.typing import NDArray
import torch

import ramannoodle as rn

from ramannoodle.pmodel.torch._gnn import PotGNN
from ramannoodle.pmodel.torch._utils import (
    _radius_graph_pbc,
    get_rotations,
)
from ramannoodle.pmodel.torch._train import train_single_epoch
from ramannoodle.dataset.torch.utils import polarizability_vectors_to_tensors


# import ramannoodle.io.vasp as vasp_io
# from ramannoodle.structure.structure_utils import apply_pbc
from ramannoodle.structure._reference import ReferenceStructure

# pylint: disable=protected-access, too-many-arguments, not-callable


def test_get_rotations() -> None:
    """Test get_rotations (normal)."""
    unit_vector = torch.randn((40, 3))

    rotation = get_rotations(unit_vector)

    rotated = torch.matmul(rotation, torch.tensor([1.0, 0.0, 0.0]))
    known = unit_vector / torch.linalg.norm(unit_vector, dim=1).view(-1, 1)

    assert torch.allclose(rotated, known, atol=1e-5)


def test_radius_graph_pbc() -> None:
    """Test _radius_graph_pbc (normal)."""
    for batch_size in range(1, 4):
        # Generate random data.
        num_atoms = 40
        lattice = torch.eye(3) * 10
        lattice = lattice.expand(batch_size, 3, 3)
        positions = torch.randn(batch_size, num_atoms, 3)

        # Batched graph
        batch_edge_index, batch_unit_vector, batch_distance = _radius_graph_pbc(
            lattice, positions, cutoff=3
        )

        # Individual graphs concatenated together
        edge_index = []
        unit_vector = []
        distance = []
        for i in range(batch_size):
            ei, uv, d = _radius_graph_pbc(
                lattice[i : i + 1], positions[i : i + 1], cutoff=3
            )
            ei[0] = i
            ei[[1, 2]] += num_atoms * i
            edge_index.append(ei)
            unit_vector.append(uv)
            distance.append(d)

        assert torch.allclose(batch_edge_index, torch.concat(edge_index, dim=1))
        assert torch.allclose(batch_unit_vector, torch.concat(unit_vector, dim=0))
        assert torch.allclose(batch_distance, torch.concat(distance, dim=0))


@pytest.mark.parametrize(
    "poscar_ref_structure_fixture",
    [
        ("test/data/TiO2/POSCAR"),
    ],
    indirect=["poscar_ref_structure_fixture"],
)
def test_batch_polarizability(poscar_ref_structure_fixture: ReferenceStructure) -> None:
    """Test of batch functions for forward pass (normal)."""
    ref_structure = poscar_ref_structure_fixture
    model = PotGNN(ref_structure, 5, 5, 5, 5, 0, 5, np.zeros((3, 3)), np.zeros((3, 3)))
    model.eval()

    for batch_size in range(1, 4):

        # Generate random data.
        num_atoms = len(ref_structure.atomic_numbers)
        lattice = torch.from_numpy(ref_structure.lattice).float()
        atomic_numbers = torch.tensor(ref_structure.atomic_numbers)

        batch_lattices = lattice.expand(batch_size, 3, 3)
        batch_atomic_numbers = atomic_numbers.expand(batch_size, num_atoms)
        batch_positions = torch.randn(batch_size, num_atoms, 3)
        batch_polarizability = model.forward(
            batch_lattices, batch_atomic_numbers, batch_positions
        )

        # Individual calls
        polarizabilities = torch.zeros((batch_size, 6))
        for i in range(batch_size):
            polarizability = model.forward(
                batch_lattices[i : i + 1],
                batch_atomic_numbers[i : i + 1],
                batch_positions[i : i + 1],
            )
            polarizabilities[i] = polarizability[0]

        assert torch.allclose(batch_polarizability, polarizabilities, atol=1e-5)


def test_reset_parameters() -> None:
    """Test of reset_parameters (normal)."""
    ref_structure = rn.io.vasp.poscar.read_ref_structure("test/data/TiO2/POSCAR")
    model = PotGNN(ref_structure, 5, 5, 5, 5, 0, 5, np.zeros((3, 3)), np.zeros((3, 3)))
    model.reset_parameters()


@pytest.mark.parametrize(
    "poscar_ref_structure_fixture",
    [
        ("test/data/TiO2/POSCAR"),
    ],
    indirect=["poscar_ref_structure_fixture"],
)
def test_gpu(poscar_ref_structure_fixture: ReferenceStructure) -> None:
    """Test putting model on gpus."""
    ref_structure = poscar_ref_structure_fixture

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    # mps backend doesn't work with github runners
    # https://github.com/actions/runner-images/issues/9918
    elif torch.backends.mps.is_available() and os.getenv("GITHUB_ACTIONS") != "true":
        device = "mps"
    else:
        return

    torch.set_default_device(device)  # type: ignore
    model = PotGNN(ref_structure, 5, 5, 5, 5, 0, 5, np.zeros((3, 3)), np.ones((3, 3)))
    model.eval()

    for batch_size in range(1, 4):

        # Generate random data.
        num_atoms = len(ref_structure.atomic_numbers)
        lattice = torch.from_numpy(ref_structure.lattice).float().to("mps")
        atomic_numbers = torch.tensor(ref_structure.atomic_numbers)

        batch_lattices = lattice.expand(batch_size, 3, 3)
        batch_atomic_numbers = atomic_numbers.expand(batch_size, num_atoms)
        batch_positions = torch.randn(batch_size, num_atoms, 3)
        _ = model.forward(batch_lattices, batch_atomic_numbers, batch_positions)

    torch.set_default_device("cpu")  # type: ignore


@pytest.mark.parametrize(
    "poscar_ref_structure_fixture",
    [
        ("test/data/TiO2/POSCAR"),
    ],
    indirect=["poscar_ref_structure_fixture"],
)
def test_calc_polarizabilities(
    poscar_ref_structure_fixture: ReferenceStructure,
) -> None:
    """Test of calc_polarizabilities (normal)."""
    ref_structure = poscar_ref_structure_fixture
    model = PotGNN(ref_structure, 2, 5, 5, 5, 0, 5, np.zeros((3, 3)), np.ones((3, 3)))
    model.eval()

    for batch_size in [50, 100, 180]:

        # Generate random data.
        num_atoms = len(ref_structure.atomic_numbers)
        lattice = torch.from_numpy(ref_structure.lattice).float()
        atomic_numbers = torch.tensor(ref_structure.atomic_numbers)

        batch_lattices = lattice.expand(batch_size, 3, 3)
        batch_atomic_numbers = atomic_numbers.expand(batch_size, num_atoms)
        batch_positions = torch.randn(batch_size, num_atoms, 3)

        forward = model.forward(batch_lattices, batch_atomic_numbers, batch_positions)
        forward = polarizability_vectors_to_tensors(forward.detach().clone()).numpy()
        calc = model.calc_polarizabilities(batch_positions.detach().clone().numpy())

        assert np.allclose(forward, calc, atol=1e-6)


@pytest.mark.parametrize(
    "cutoff,size_node_embedding,size_edge_embedding,num_message_passes,"
    "gaussian_filter_start,gaussian_filter_end,mean_polarizability,"
    "stddev_polarizability,exception_type,in_reason",
    [
        (
            0,
            5,
            5,
            5,
            0,
            5,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            ValueError,
            "invalid cutoff: 0 <= 0",
        ),
        (
            5,
            0,
            5,
            5,
            0,
            5,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            ValueError,
            "invalid size_node_embedding: 0 <= 0",
        ),
        (
            5,
            5,
            0,
            5,
            0,
            5,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            ValueError,
            "invalid size_edge_embedding: 0 <= 0",
        ),
        (
            5,
            5,
            5,
            0,
            0,
            5,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            ValueError,
            "invalid num_message_passes: 0 <= 0",
        ),
        (
            5,
            5,
            5,
            5,
            -1,
            5,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            ValueError,
            "invalid gaussian_filter_start: -1 < 0",
        ),
        (
            5,
            5,
            5,
            5,
            5,
            0,
            np.zeros((3, 3)),
            np.zeros((3, 3)),
            ValueError,
            "invalid gaussian_filter_end: 0 <= gaussian_filter_start",
        ),
    ],
)
# pylint: disable=too-many-arguments,too-many-positional-arguments
def test_gnn_exception(
    cutoff: float,
    size_node_embedding: int,
    size_edge_embedding: int,
    num_message_passes: int,
    gaussian_filter_start: float,
    gaussian_filter_end: float,
    mean_polarizability: NDArray[np.float64],
    stddev_polarizability: NDArray[np.float64],
    exception_type: Type[Exception],
    in_reason: str,
) -> None:
    """Test gnn (exception)."""
    ref_structure = rn.io.vasp.poscar.read_ref_structure("test/data/TiO2/POSCAR")
    with pytest.raises(exception_type, match=re.escape(in_reason)):
        _ = PotGNN(
            ref_structure,
            cutoff,
            size_node_embedding,
            size_edge_embedding,
            num_message_passes,
            gaussian_filter_start,
            gaussian_filter_end,
            mean_polarizability,
            stddev_polarizability,
        )


def test_train_single_epoch() -> None:
    """Test train_single_epoch."""
    ref_structure = rn.io.vasp.poscar.read_ref_structure("test/data/TiO2/POSCAR")
    model = PotGNN(ref_structure, 2, 5, 14, 2, 0, 5, np.zeros((3, 3)), np.zeros((3, 3)))
    test_set = rn.io.generic.read_polarizability_dataset(
        filepaths=[
            "test/data/TiO2/O43_0.1x_eps_OUTCAR",
            "test/data/TiO2/O43_0.1y_eps_OUTCAR",
        ],
        file_format="outcar",
    )
    validation_set = rn.io.generic.read_polarizability_dataset(
        filepaths=[
            "test/data/TiO2/O43_m0.1x_eps_OUTCAR",
            "test/data/TiO2/O43_m0.1y_eps_OUTCAR",
        ],
        file_format="outcar",
    )

    batch_size = 1
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # type: ignore
    loss_function = torch.nn.MSELoss()

    _ = train_single_epoch(
        model,
        test_set,
        validation_set,
        batch_size,
        optimizer,
        loss_function,
    )


# @pytest.mark.parametrize(
#     "poscar_ref_structure_fixture",
#     [
#         ("test/data/TiO2/POSCAR"),
#         ("test/data/Ag2Mo2O7.poscar"),
#         ("test/data/STO/SrTiO3.poscar"),
#         ("test/data/SbPC3S3N3Cl3O.poscar"),
#     ],
#     indirect=["poscar_ref_structure_fixture"],
# )
# def test_symmetry(poscar_ref_structure_fixture: ReferenceStructure) -> None:
#     """Test that model obeys symmetries."""
#     ref_structure = poscar_ref_structure_fixture
#     model = PotGNN(ref_structure, 2.3, 5, 5, 6, np.zeros((3, 3)), np.zeros((3, 3)))
#     model.eval()

#     lattice = torch.tensor([ref_structure.lattice]).float()
#     atomic_numbers = torch.tensor([ref_structure.atomic_numbers], dtype=torch.int)
#     positions = torch.tensor([ref_structure.positions]).float()

#     polarizability = model.forward(lattice, atomic_numbers, positions)
#     polarizability = polarizability_vectors_to_tensors(polarizability)
#     polarizability = polarizability[0].detach().numpy()

#     assert ref_structure._symmetry_dict is not None
#     for rotation in ref_structure._symmetry_dict["rotations"]:
#         rotated_polarizability = np.linalg.inv(rotation) @ polarizability @ rotation

#         assert np.allclose(polarizability, rotated_polarizability, atol=1e-5)


# def test_symmetry_displaced() -> None:
#     """Test that model obeys symmetries."""
#     ref_structure = vasp_io.poscar.read_ref_structure("test/data/TiO2/POSCAR")
#     displaced_positions = vasp_io.outcar.read_positions(
#         "test/data/TiO2/Ti5_0.1x_eps_OUTCAR"
#     )
#     model = PotGNN(ref_structure, 5, 5, 6, 4, np.zeros((3, 3)), np.zeros((3, 3)))
#     model.eval()
#     parent_displacement = (displaced_positions - ref_structure.positions) / (
#         (np.linalg.norm(displaced_positions - ref_structure.positions) * 10)
#     )

#     lattice = torch.tensor([ref_structure.lattice]).float()
#     atomic_numbers = torch.tensor([ref_structure.atomic_numbers], dtype=torch.int)
#     positions = torch.tensor([ref_structure.positions + parent_displacement]).float()

#     polarizability = model.forward(lattice, atomic_numbers, positions)
#     polarizability = polarizability_vectors_to_tensors(polarizability)
#     polarizability = polarizability[0].detach().numpy()

#     displacements_and_transformations = ref_structure.get_equivalent_displacements(
#         parent_displacement
#     )
#     for dof_dictionary in displacements_and_transformations:
#         for displacement, transformation in zip(
#             dof_dictionary["displacements"], dof_dictionary["transformations"]
#         ):
#             rotation = transformation[0]
#             rotated_polarizability = (
#               np.linalg.inv(rotation) @ polarizability @ rotation
#             )
#
#             positions = torch.tensor(
#                 [apply_pbc(ref_structure.positions + displacement)]
#             ).float()
#             model_polarizability = model.forward(lattice, atomic_numbers, positions)
#             model_polarizability = polarizability_vectors_to_tensors(
#                 model_polarizability
#             )
#             model_polarizability = model_polarizability[0].detach().numpy()

#             assert np.isclose(
#                 rotated_polarizability, model_polarizability, atol=1e-5
#             ).all()
