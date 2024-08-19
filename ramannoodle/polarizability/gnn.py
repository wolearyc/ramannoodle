"""Polarizability model based on a graph neural network (GNN)."""

# pylint: disable=not-callable

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor
from torch.nn import BatchNorm1d, Embedding, Linear, ModuleList, Sequential
from torch.utils.data import Dataset

from torch_geometric.nn.inits import reset
from torch_geometric.nn.models.dimenet import triplets
from torch_geometric.nn.models.schnet import ShiftedSoftplus
from torch_geometric.typing import OptTensor
from torch_geometric.utils import scatter


class PolarizabilityDataset(Dataset):
    """Pytorch dataset containing atom positions mapped to polarizabilities.

    Polarizabilities are scaled and reshaped into tensors with size [6,].

    Parameters
    ----------
    positions
        Unitless | 3D array with shape (S,N,3) where N is the number of atoms and S is
        the number of samples.
    lattices
        Å | 3D array with shape (S,3,3)
    polarizabilities
        Unitless | 3D array with shape (S,3,3).

    """

    def __init__(
        self,
        lattices: NDArray[np.float64],
        atomic_numbers: list[list[int]],
        positions: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
    ):
        self._lattices = torch.from_numpy(lattices).float()
        self._atomic_numbers = torch.tensor(atomic_numbers)
        self._positions = torch.from_numpy(positions).float()
        self._polarizabilities = torch.from_numpy(polarizabilities).float()

        self._mean_polarizability = self._polarizabilities.mean(0, keepdim=True)
        self._stddev_polarizability = self._polarizabilities.std(
            0, unbiased=False, keepdim=True
        )
        scaled_polarizabilities = (
            self._polarizabilities - self._mean_polarizability
        ) / self._stddev_polarizability

        self._scaled_polarizabilities = torch.zeros(
            (scaled_polarizabilities.size()[0], 6)
        )

        self._scaled_polarizabilities[:, 0] = scaled_polarizabilities[:, 0, 0]
        self._scaled_polarizabilities[:, 1] = scaled_polarizabilities[:, 1, 1]
        self._scaled_polarizabilities[:, 2] = scaled_polarizabilities[:, 2, 2]
        self._scaled_polarizabilities[:, 3] = scaled_polarizabilities[:, 0, 1]
        self._scaled_polarizabilities[:, 4] = scaled_polarizabilities[:, 0, 2]
        self._scaled_polarizabilities[:, 5] = scaled_polarizabilities[:, 1, 2]

    def __len__(self) -> int:
        """Get length."""
        return len(self._positions)

    def __getitem__(self, i):
        """Get positions, atomic numbers, lattices, and scaled polarizabilities."""
        return (
            self._lattices[i],
            self._atomic_numbers[i],
            self._positions[i],
            self._scaled_polarizabilities[i],
        )


class GaussianFilter(torch.nn.Module):
    """Gaussian filter."""

    def __init__(self, start: float, stop: float, steps: int):
        super().__init__()
        offset = torch.linspace(start, stop, steps)
        self.coefficient = -0.5 / (float(offset[1]) - float(offset[0])) ** 2
        self.register_buffer("offset", offset)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        X
            Tensor with size []. Typically contains interatomic distances.

        """
        x = x.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coefficient * x.pow(2))


class NodeBlock(torch.nn.Module):
    """Edge to node message passer.

    Architecture and notation is based on equation (5) in https://doi.org/10.1038/
    s41524-021-00543-3. The architecture has been modified to use batch normalization.

    Parameters
    ----------
    hidden_node_channels
    hidden_edge_channels
    """

    def __init__(self, hidden_node_channels: int, hidden_edge_channels: int):
        super().__init__()

        # Combination of two linear layers, dubbed "core" and "filter":
        #   "filter" : c1ij W1 + b1
        #   "core"   : c1ij W2 + b2
        self.lin_c1 = Linear(
            hidden_node_channels + hidden_edge_channels, 2 * hidden_node_channels
        )

        self.bn_c1 = BatchNorm1d(2 * hidden_node_channels)
        self.bn = BatchNorm1d(hidden_node_channels)

    def reset_parameters(self):
        """Reset model parameters."""
        self.lin_c1.reset_parameters()
        self.bn_c1.reset_parameters()
        self.bn.reset_parameters()

    def forward(
        self, node_embedding: Tensor, edge_embedding: Tensor, i: Tensor
    ) -> Tensor:
        """Forward pass."""
        c1 = torch.cat([node_embedding[i], edge_embedding], dim=1)
        c1 = self.bn_c1(self.lin_c1(c1))
        c1_filter, c1_core = c1.chunk(2, dim=1)
        c1_filter = c1_filter.sigmoid()
        c1_core = c1_core.tanh()
        c1_emb = scatter(
            c1_filter * c1_core, i, dim=0, dim_size=node_embedding.size(0), reduce="sum"
        )
        c1_emb = self.bn(c1_emb)

        return (node_embedding + c1_emb).tanh()


class EdgeBlock(torch.nn.Module):
    """Node to edge message passer.

    Architecture and notation is based on equation (6) in https://doi.org/10.1038/
    s41524-021-00543-3. The architecture has been modified to use batch normalization.

    Parameters
    ----------
    hidden_node_channels
    hidden_edge_channels
    """

    def __init__(self, hidden_node_channels: int, hidden_edge_channels: int):
        super().__init__()

        # Combination of two linear layers, dubbed "core" and "filter":
        #   "filter" : c2ij W3 + b3
        #   "core"   : c2ij W4 + b4
        self.lin_c2 = Linear(hidden_node_channels, 2 * hidden_edge_channels)

        # Combination of two linear layers, dubbed "core" and "filter":
        #   "filter" : c3ij W5 + b5
        #   "core"   : c3ij W6 + b6
        self.lin_c3 = Linear(
            3 * hidden_node_channels + 2 * hidden_edge_channels,
            2 * hidden_edge_channels,
        )

        self.bn_c2 = BatchNorm1d(2 * hidden_edge_channels)
        self.bn_c3 = BatchNorm1d(2 * hidden_edge_channels)
        self.bn_c2_2 = BatchNorm1d(hidden_edge_channels)
        self.bn_c3_2 = BatchNorm1d(hidden_edge_channels)

    def reset_parameters(self):
        """Reset model parameters."""
        self.lin_c2.reset_parameters()
        self.lin_c3.reset_parameters()
        self.bn_c2.reset_parameters()
        self.bn_c3.reset_parameters()
        self.bn_c2_2.reset_parameters()
        self.bn_c3_2.reset_parameters()

    def forward(  # pylint: disable=too-many-arguments, too-many-locals
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        i: Tensor,
        j: Tensor,
        idx_i: Tensor,
        idx_j: Tensor,
        idx_k: Tensor,
        idx_ji: Tensor,
        idx_kj: Tensor,
    ) -> Tensor:
        """Forward pass."""
        c2 = node_embedding[i] * node_embedding[j]
        c2 = self.bn_c2(self.lin_c2(c2))
        c2_filter, c2_core = c2.chunk(2, dim=1)
        c2_filter = c2_filter.sigmoid()
        c2_core = c2_core.tanh()
        c2_emb = self.bn_c2_2(c2_filter * c2_core)

        c3 = torch.cat(
            [
                node_embedding[idx_i],
                node_embedding[idx_j],
                node_embedding[idx_k],
                edge_embedding[idx_ji],
                edge_embedding[idx_kj],
            ],
            dim=1,
        )
        c3 = self.bn_c3(self.lin_c3(c3))
        c3_filter, c3_core = c3.chunk(2, dim=1)
        c3_filter = c3_filter.sigmoid()
        c3_core = c3_core.tanh()
        c3_emb = scatter(
            c3_filter * c3_core,
            idx_ji,
            dim=0,
            dim_size=edge_embedding.size(0),
            reduce="sum",
        )
        c3_emb = self.bn_c3_2(c3_emb)

        return (edge_embedding + c2_emb + c3_emb).tanh()


class PotGNN(torch.nn.Module):
    r"""POlarizability Tensor Graph Neural Network (PotGNN).

    GNN architecture was inspired by the "direct force architecture" developed in Park
    `et al.`;  `npj Computational Materials` (2021)7:73; https://doi.org/10.1038/
    s41524-021-00543-3. Implementation adapted from ``torch_geometric.nn.models.GNNFF``
    authored by @ken2403 and merged by @rusty1s.

    Parameters
    ----------
    hidden_node_channels
        Hidden node embedding size.
    hidden_edge_channels
        Hidden edge embedding size.
    num_layers
        Number of message passing blocks.
    cutoff
        Cutoff distance for interatomic interactions.
    """

    def __init__(
        self,
        hidden_node_channels: int,
        hidden_edge_channels: int,
        num_layers: int,
        cutoff: float = 5.0,
    ):
        super().__init__()

        self.cutoff = cutoff

        self.node_embedding = Sequential(
            Embedding(95, hidden_node_channels),
            ShiftedSoftplus(),  # nonlinear activation layer
            Linear(hidden_node_channels, hidden_node_channels),
            ShiftedSoftplus(),  # nonlinear activation layer
            Linear(hidden_node_channels, hidden_node_channels),
        )
        self.edge_embedding = GaussianFilter(0.0, 5.0, hidden_edge_channels)

        self.node_blocks = ModuleList(
            [
                NodeBlock(hidden_node_channels, hidden_edge_channels)
                for _ in range(num_layers)
            ]
        )
        self.edge_blocks = ModuleList(
            [
                EdgeBlock(hidden_node_channels, hidden_edge_channels)
                for _ in range(num_layers)
            ]
        )

        self.polarizability_predictor = Sequential(
            Linear(hidden_edge_channels, hidden_edge_channels),
            ShiftedSoftplus(),
            Linear(hidden_edge_channels, hidden_edge_channels),
            ShiftedSoftplus(),
            Linear(hidden_edge_channels, 6),
        )

    def reset_parameters(self):
        """Reset model parameters."""
        reset(self.node_embedding)
        self.edge_embedding.reset_parameters()
        for node_block in self.node_blocks:
            node_block.reset_parameters()
        for edge_block in self.edge_blocks:
            edge_block.reset_parameters()
        reset(self.polarizability_predictor)

    def forward(  # pylint: disable=too-many-locals
        self,
        lattice: Tensor,
        atomic_numbers: Tensor,
        positions: Tensor,
        # batch: OptTensor = None,
    ) -> Tensor:
        """Forward pass."""
        edge_index, unit_vec, dist = _radius_graph_pbc(lattice, positions, self.cutoff)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index, num_nodes=atomic_numbers.size(0)
        )

        # Calculate distances and unit vector:
        # dist = (positions[i] - positions[j]).pow(2).sum(dim=-1).sqrt()
        # unit_vec = (positions[i] - positions[j]) / dist.view(-1, 1)

        # Embedding blocks:
        node_emb = self.node_embedding(atomic_numbers)
        edge_emb = self.edge_embedding(dist)

        # Message passing blocks:
        for node_block, edge_block in zip(self.node_blocks, self.edge_blocks):
            node_emb = node_block(node_emb, edge_emb, i)
            edge_emb = edge_block(
                node_emb, edge_emb, i, j, idx_i, idx_j, idx_k, idx_ji, idx_kj
            )

        # Polarizability prediction block:
        polarizability = self.polarizability_predictor(edge_emb)
        polarizability = self._get_polarizability_tensors(polarizability)
        rotation = self._get_rotation_matrices(unit_vec)

        polarizability = torch.matmul(rotation, polarizability)
        polarizability = torch.matmul(polarizability, torch.linalg.inv(rotation))

        polarizability = self._get_polarizability_vectors(polarizability)

        return torch.mean(polarizability, dim=0)

    def _get_polarizability_tensors(self, x: Tensor) -> Tensor:
        """X should have size (_,6)."""
        indices = torch.tensor(
            [
                [0, 3, 4],
                [3, 1, 5],
                [4, 5, 2],
            ]
        )
        return x[:, indices]

    def _get_polarizability_vectors(self, x: Tensor) -> Tensor:
        """X should have size (_,3,3)."""
        indices = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]).T
        return x[:, indices[0], indices[1]]

    def _get_rotation_matrices(self, destination: Tensor):
        """Get rotation matrices.

        The source vector is set to (1,0,0). I think this should be in cartesian??

        Parameters
        ----------
        destination
            Destination vector. Expected to be list of unit vectors.
        """
        source = torch.zeros(destination.size())  # source vector
        source[:, 0] = 1

        # Normalize all vectors
        a = source / torch.linalg.norm(source, dim=1).view(-1, 1)
        b = destination / torch.linalg.norm(destination, dim=1).view(-1, 1)

        v = torch.linalg.cross(a, b)
        c = torch.linalg.vecdot(a, b)
        s = torch.linalg.norm(v, dim=1)

        kmat = torch.zeros((len(v), 3, 3))
        kmat[:, 0, 1] = -v[:, 2]
        kmat[:, 0, 2] = v[:, 1]
        kmat[:, 1, 0] = v[:, 2]
        kmat[:, 1, 2] = -v[:, 0]
        kmat[:, 2, 0] = -v[:, 1]
        kmat[:, 2, 1] = v[:, 0]

        eye_matrix = torch.zeros((len(v), 3, 3))
        eye_matrix[:] = torch.eye(3)
        rotation_matrix = eye_matrix
        rotation_matrix += kmat
        a1 = kmat.matmul(kmat)
        b1 = (1 - c) / (s**2)
        rotation_matrix += a1 * b1[:, None, None]
        return rotation_matrix


def _radius_graph_pbc(lattice: Tensor, positions: Tensor, cutoff: float):
    """Generate edge indexes and unit vectors."""
    # Compute pairwise distance matrix.
    positions_a = positions.unsqueeze(1)
    positions_b = positions.unsqueeze(0)
    displacement = positions_b - positions_a
    displacement = torch.where(
        displacement % 1 > 0.5, displacement % 1 - 1, displacement % 1
    )
    cart_displacement = displacement.matmul(lattice)
    cart_distance_matrix = torch.sqrt(torch.sum(cart_displacement**2, dim=-1))

    # Valid edges
    valid_edge = cart_distance_matrix <= cutoff
    not_self_loop = ~torch.eye(valid_edge.size()[0], dtype=torch.bool)
    valid_edges = torch.logical_and(valid_edge, not_self_loop)

    # Edge indexes and unit vectors
    edge_indexes = torch.nonzero(valid_edges).T
    cart_unit_vectors = cart_displacement[edge_indexes[0], edge_indexes[1]]
    cart_unit_vectors /= torch.linalg.norm(cart_unit_vectors, dim=1)[:, None]
    distances = cart_distance_matrix[edge_indexes[0], edge_indexes[1]].view(-1, 1)
    return edge_indexes, cart_unit_vectors, distances


# def test_get_rotation_matrices():
#    destination = torch.randn((30,3))
#    mat = model._get_rotation_matrices(destination)
#    vec1_rot = mat.matmul(torch.tensor([1.0, 0.0, 0.0]))
#
#    rotated = vec1_rot
#    known = destination / torch.linalg.norm(destination, dim=1).view(-1, 1)
#
#    assert torch.allclose(rotated, known, atol = 1e-5)
#    print("Success!")
# test_get_rotation_matrices()
