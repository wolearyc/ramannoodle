"""Testing for GNN-based models."""

import torch

from ramannoodle.polarizability.gnn import PotGNN, _radius_graph_pbc

# pylint: disable=protected-access, too-many-arguments, not-callable


def test_get_rotations() -> None:
    """Test _get_rotations (normal)."""
    model = PotGNN(5, 5, 5, 5)
    unit_vector = torch.randn((40, 3))

    rotation = model._get_rotations(unit_vector)
    rotated = rotation.matmul(torch.tensor([1.0, 0.0, 0.0]))
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


def test_batch_polarizability() -> None:
    """Test of batch functions for forward pass (normal)."""
    for batch_size in range(1, 4):
        model = PotGNN(5, 5, 5, 5)
        model.eval()

        # Generate random data.
        num_atoms = 40
        lattice = torch.eye(3) * 10
        atomic_numbers = torch.randint(1, 10, (num_atoms,))

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

        assert torch.allclose(batch_polarizability, polarizabilities)
