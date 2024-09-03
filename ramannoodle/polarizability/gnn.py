"""Polarizability model based on a graph neural network (GNN)."""

# pylint: disable=not-callable

from __future__ import annotations

import typing
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    Sequential,
    Module,
    LayerNorm,
)
from torch.utils.data import Dataset

from torch_geometric.nn.inits import reset
from torch_geometric.nn.models.dimenet import triplets
from torch_geometric.nn.models.schnet import ShiftedSoftplus
from torch_geometric.utils import scatter

from ramannoodle.structure.reference import ReferenceStructure
from ramannoodle.exceptions import get_type_error


def _get_tensor_size_str(size: Sequence[int | None]) -> str:
    """Get a string representing a tensor size.

    Maps None --> "_", indicating that this dimension can be any size.
    """
    result = "["
    for i in size:
        if i is None:
            result += "_,"
        else:
            result += f"{i},"
    if len(size) == 1:
        return result + "]"
    return result[:-1] + "]"


def _get_size_error_tensor(name: str, tensor: Tensor, desired_size: str) -> ValueError:
    """Return ValueError for an pytorch tensor with the wrong size."""
    shape_spec = f"{_get_tensor_size_str(tensor.size())} != {desired_size}"
    return ValueError(f"{name} has wrong size: {shape_spec}")


def _get_scaled_polarizabilities(
    polarizabilities: Tensor,
    scale: str,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute standard-scaled and flattened (6 member) polarizabilities.

    Parameters
    ----------
    polarizabilities
        Tensor with size [S,3,3] where S is the number of samples.

    Returns
    -------
    :
        3-tuple. The first element is the element-wise mean of polarizabilities, a
        tensor with size [1,3,3], The second element is the maximum standard deviation,
        a tensor with size [1,]. The third element is the scaled and flattened
        polarizabilities, a tensor with size [S,6].

    """
    mean = polarizabilities.mean(0, keepdim=True)
    stddev = polarizabilities.std(0, unbiased=False, keepdim=True)
    if scale == "standard":
        polarizabilities = (polarizabilities - mean) / stddev
    elif scale == "stddev":
        polarizabilities = (polarizabilities - mean) / stddev + mean
    elif scale != "none":
        raise ValueError("invalid scale option")

    scaled_polarizabilities = torch.zeros((polarizabilities.size(0), 6))
    scaled_polarizabilities[:, 0] = polarizabilities[:, 0, 0]
    scaled_polarizabilities[:, 1] = polarizabilities[:, 1, 1]
    scaled_polarizabilities[:, 2] = polarizabilities[:, 2, 2]
    scaled_polarizabilities[:, 3] = polarizabilities[:, 0, 1]
    scaled_polarizabilities[:, 4] = polarizabilities[:, 0, 2]
    scaled_polarizabilities[:, 5] = polarizabilities[:, 1, 2]

    return mean, stddev, scaled_polarizabilities


def _get_rotations(targets: Tensor) -> Tensor:
    """Get rotation matrices to target vectors from a reference vector (1,0,0).

    Parameters
    ----------
    targets
        Tensor with size [S,3]. Vector components do not need to be normalized.

    Returns
    -------
    :
        Tensor with size [S,3,3].
    """
    reference = torch.zeros(targets.size())
    reference[:, 0] = 1

    a = reference / torch.linalg.norm(reference, dim=1).view(-1, 1)
    b = targets / torch.linalg.norm(targets, dim=1).view(-1, 1)

    v = torch.linalg.cross(a, b)  # This will be (0,0,0) if a == b
    c = torch.linalg.vecdot(a, b)
    s = torch.linalg.norm(v, dim=1)  # This will be zero if a == b

    k_matrix = torch.zeros((len(v), 3, 3))
    k_matrix[:, 0, 1] = -v[:, 2]
    k_matrix[:, 0, 2] = v[:, 1]
    k_matrix[:, 1, 0] = v[:, 2]
    k_matrix[:, 1, 2] = -v[:, 0]
    k_matrix[:, 2, 0] = -v[:, 1]
    k_matrix[:, 2, 1] = v[:, 0]

    rotations = torch.zeros((len(v), 3, 3))
    rotations[:] = torch.eye(3)  # Rotation starts as identity
    rotations += k_matrix
    a1 = k_matrix.matmul(k_matrix)
    b1 = (1 - c) / (s**2)
    rotations += a1 * b1[:, None, None]

    # a==b implies s==0 implies rotation should be identity
    rotations[s == 0] = torch.eye(3)
    return rotations


def _get_atom_types(atomic_numbers: list[list[int]]) -> Tensor:
    """Convert atomic numbers into a one-hot encoding."""
    ndarray = np.array(atomic_numbers)
    unique_atomic_numbers = set(ndarray.flatten())

    for i, atomic_number in enumerate(unique_atomic_numbers):
        ndarray[ndarray == atomic_number] = i
    return torch.tensor(ndarray)


class PolarizabilityDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """PyTorch dataset of atomic structures and polarizabilities.

    Polarizabilities are internally scaled and flattened, leaving polarizability
    vectors containing the six independent tensor components.

    Parameters
    ----------
    lattices
        Å | 3D array with shape (S,3,3)
    atomic_numbers
        List of lists of length N, where N is the number of atoms.
    positions
        Unitless | 3D array with shape (S,N,3) where S is the number of samples.
    polarizabilities
        Unitless | 3D array with shape (S,3,3).

    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        lattices: NDArray[np.float64],
        atomic_numbers: list[list[int]],
        positions: NDArray[np.float64],
        polarizabilities: NDArray[np.float64],
        scale: str = "none",
    ):
        default_type = torch.get_default_dtype()
        self._lattices = torch.from_numpy(lattices).type(default_type)
        self._atomic_numbers = torch.tensor(atomic_numbers)
        self._positions = torch.from_numpy(positions).type(default_type)
        mean, stddev, scaled = _get_scaled_polarizabilities(
            torch.from_numpy(polarizabilities), scale=scale
        )
        self._mean_polarizability = mean.type(default_type)
        self._stddev_polarizability = stddev.type(default_type)
        self._polarizabilities = scaled.type(default_type)

    def __len__(self) -> int:
        """Get length."""
        return len(self._positions)

    def __getitem__(self, i: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Get lattice, atomic numbers, positions, and polarizabilities."""
        return (
            self._lattices[i],
            self._atomic_numbers[i],
            self._positions[i],
            self._polarizabilities[i],
        )


class GaussianFilter(torch.nn.Module):
    """Gaussian filter.

    Parameters should be chosen such that all expected inputs are between start and
    stop.

    Parameters
    ----------
    start
        Lower bound for filter input.
    stop
        Upper bound for filter input.
    steps
        Number of steps between start and stop.
    """

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
            Typically contains interatomic distances.

        """
        x = x.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coefficient * x.pow(2))


class NodeBlock(torch.nn.Module):
    """Edge to node message passer.

    Architecture and notation is based on equation (5) in https://doi.org/10.1038/
    s41524-021-00543-3. The architecture has been modified to use batch normalization.
    """

    def __init__(
        self,
        size_node_embedding: int,
        size_edge_embedding: int,
    ):
        super().__init__()

        # Combination of two linear layers, dubbed "core" and "filter":
        #   "filter" : c1ij W1 + b1
        #   "core"   : c1ij W2 + b2
        self.lin_c1 = Linear(
            size_node_embedding + size_edge_embedding,
            2 * size_node_embedding,
        )

        self.bn_c1 = LayerNorm(2 * size_node_embedding)
        self.bn = LayerNorm(size_node_embedding)

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        self.lin_c1.reset_parameters()
        self.bn_c1.reset_parameters()
        self.bn.reset_parameters()

    def forward(
        self, node_embedding: Tensor, edge_embedding: Tensor, i: Tensor
    ) -> Tensor:
        """Forward pass.

        (Node embedding, edge embedding) -> node embedding.
        """
        c1 = torch.cat([node_embedding[i], edge_embedding], dim=1)
        c1 = self.bn_c1(self.lin_c1(c1))
        c1_filter, c1_core = c1.chunk(2, dim=1)
        c1_filter = c1_filter.sigmoid()
        c1_core = c1_core.tanh()
        c1_emb = scatter(
            c1_filter * c1_core, i, dim=0, dim_size=node_embedding.size(0), reduce="sum"
        )
        c1_emb = self.bn(c1_emb)

        return typing.cast(Tensor, (node_embedding + c1_emb).tanh())


class EdgeBlock(torch.nn.Module):
    """Node to edge message passer.

    Architecture and notation is based on equation (6) in https://doi.org/10.1038/
    s41524-021-00543-3. The architecture has been modified to use batch normalization.
    """

    def __init__(
        self,
        size_node_embedding: int,
        size_edge_embedding: int,
    ):
        super().__init__()

        # Combination of two linear layers, dubbed "core" and "filter":
        #   "filter" : c2ij W3 + b3
        #   "core"   : c2ij W4 + b4
        self.lin_c2 = Linear(size_node_embedding, 2 * size_edge_embedding)

        # Combination of two linear layers, dubbed "core" and "filter":
        #   "filter" : c3ij W5 + b5
        #   "core"   : c3ij W6 + b6
        self.lin_c3 = Linear(
            3 * size_node_embedding + 2 * size_edge_embedding,
            2 * size_edge_embedding,
        )

        self.bn_c2 = LayerNorm(2 * size_edge_embedding)
        self.bn_c3 = LayerNorm(2 * size_edge_embedding)
        self.bn_c2_2 = LayerNorm(size_edge_embedding)
        self.bn_c3_2 = LayerNorm(size_edge_embedding)

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        self.lin_c2.reset_parameters()
        self.lin_c3.reset_parameters()
        self.bn_c2.reset_parameters()
        self.bn_c3.reset_parameters()
        self.bn_c2_2.reset_parameters()
        self.bn_c3_2.reset_parameters()

    def _get_c2_embedding(
        self,
        node_embedding: Tensor,
        i: Tensor,
        j: Tensor,
    ) -> Tensor:
        """Get c2 embedding."""
        c2 = node_embedding[i] * node_embedding[j]
        c2 = self.bn_c2(self.lin_c2(c2))
        c2_filter, c2_core = c2.chunk(2, dim=1)
        c2_filter = c2_filter.sigmoid()
        c2_core = c2_core.tanh()
        return typing.cast(Tensor, self.bn_c2_2(c2_filter * c2_core))

    def _get_c3_embedding(  # pylint: disable=too-many-arguments
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        index_i: Tensor,
        index_j: Tensor,
        index_k: Tensor,
        index_ji: Tensor,
        index_kj: Tensor,
    ) -> Tensor:
        """Get c3 embedding."""
        c3 = torch.cat(
            [
                node_embedding[index_i],
                node_embedding[index_j],
                node_embedding[index_k],
                edge_embedding[index_ji],
                edge_embedding[index_kj],
            ],
            dim=1,
        )
        c3 = self.bn_c3(self.lin_c3(c3))
        c3_filter, c3_core = c3.chunk(2, dim=1)
        c3_filter = c3_filter.sigmoid()
        c3_core = c3_core.tanh()
        c3_emb = scatter(
            c3_filter * c3_core,
            index_ji,
            dim=0,
            dim_size=edge_embedding.size(0),
            reduce="sum",
        )
        return typing.cast(Tensor, self.bn_c3_2(c3_emb))

    def forward(  # pylint: disable=too-many-arguments
        self,
        node_embedding: Tensor,
        edge_embedding: Tensor,
        i: Tensor,
        j: Tensor,
        index_i: Tensor,
        index_j: Tensor,
        index_k: Tensor,
        index_ji: Tensor,
        index_kj: Tensor,
    ) -> Tensor:
        """Forward pass."""
        c2_embedding = self._get_c2_embedding(node_embedding, i, j)
        c3_embedding = self._get_c3_embedding(
            node_embedding,
            edge_embedding,
            index_i,
            index_j,
            index_k,
            index_ji,
            index_kj,
        )
        return (edge_embedding + c2_embedding + c3_embedding).tanh()


def _polarizability_tensors_to_vectors(polarizability_tensors: Tensor) -> Tensor:
    """Convert polarizability vectors to tensors.

    Parameters
    ----------
    polarizability_vectors
        Tensor with size [S,6].

    Returns
    -------
    :
        Symmetric tensor with size [S,3,3].
    """
    indices = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]).T
    return polarizability_tensors[:, indices[0], indices[1]]


def polarizability_vectors_to_tensors(polarizability_vectors: Tensor) -> Tensor:
    """Convert polarizability vectors to tensors.

    Parameters
    ----------
    polarizability_vectors
        Tensor with size [S,6].

    Returns
    -------
    :
        Symmetric tensor with size [S,3,3].
    """
    indices = torch.tensor(
        [
            [0, 3, 4],
            [3, 1, 5],
            [4, 5, 2],
        ]
    )
    try:
        return polarizability_vectors[:, indices]
    except IndexError as exc:
        raise _get_size_error_tensor(
            "polarizability_vectors", polarizability_vectors, "[_,6]"
        ) from exc
    except TypeError as exc:
        raise get_type_error(
            "polarizability_vectors", polarizability_vectors, "Tensor"
        ) from exc


class PotGNN(Module):  # pylint: disable = too-many-instance-attributes
    r"""POlarizability Tensor Graph Neural Network (PotGNN).

    GNN architecture was inspired by the "direct force architecture" developed in Park
    `et al.`;  `npj Computational Materials` (2021)7:73; https://doi.org/10.1038/
    s41524-021-00543-3. Implementation adapted from ``torch_geometric.nn.models.GNNFF``
    authored by @ken2403 and merged by @rusty1s.

    Parameters
    ----------
    ref_structure
        Reference structure from which nodes/edges are determined.
    cutoff
        Å | Cutoff distance for interatomic interactions.
    size_node_embedding
    size_edge_embedding
    num_layers
        Number of message passing blocks.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        ref_structure: ReferenceStructure,
        cutoff: float,
        size_node_embedding: int,
        size_edge_embedding: int,
        num_message_passing_layers: int,
    ):
        super().__init__()
        default_type = torch.get_default_dtype()

        self._ref_structure = ref_structure
        self._ref_edge_indexes, _, self._ref_distances = _radius_graph_pbc(
            torch.from_numpy(ref_structure.lattice).unsqueeze(0).type(default_type),
            torch.from_numpy(ref_structure.positions).unsqueeze(0).type(default_type),
            cutoff,
        )
        self._cutoff = cutoff

        unique_atomic_numbers = set(ref_structure.atomic_numbers)
        self._num_atom_types = len(unique_atomic_numbers)
        self._atom_types = (torch.zeros(119) - 1).type(torch.int)
        for atom_type, atomic_number in enumerate(unique_atomic_numbers):
            self._atom_types[atomic_number] = atom_type

        self._node_embedding = Sequential(
            Embedding(self._num_atom_types, size_node_embedding),
            ShiftedSoftplus(),  # nonlinear activation layer
            Linear(size_node_embedding, size_node_embedding),
            ShiftedSoftplus(),  # nonlinear activation layer
            Linear(size_node_embedding, size_node_embedding),
        )
        self._edge_embedding = GaussianFilter(0.0, 5.0, size_edge_embedding)

        self._node_blocks = ModuleList(
            [
                NodeBlock(size_node_embedding, size_edge_embedding)
                for _ in range(num_message_passing_layers)
            ]
        )
        self._edge_blocks = ModuleList(
            [
                EdgeBlock(size_node_embedding, size_edge_embedding)
                for _ in range(num_message_passing_layers)
            ]
        )

        self._polarizability_predictor = Sequential(
            Linear(size_edge_embedding, size_edge_embedding),
            BatchNorm1d(size_edge_embedding),
            ShiftedSoftplus(),
            Linear(size_edge_embedding, size_edge_embedding),
            ShiftedSoftplus(),
            Linear(size_edge_embedding, 12),
        )

    def _convert_to_atom_type(self, atomic_numbers: Tensor) -> Tensor:
        return self._atom_types[atomic_numbers]

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        reset(self._node_embedding)
        self._edge_embedding.reset_parameters()
        for node_block in self._node_blocks:
            node_block.reset_parameters()
        for edge_block in self._edge_blocks:
            edge_block.reset_parameters()
        reset(self._polarizability_predictor)
        reset(self.node_polarizability_predictor)

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

    def _get_edge_polarizability_tensor(
        self, x: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """X should have size (_,2)."""
        t1 = torch.zeros((x.size(0), 3, 3))
        t1[:, 0, 0] = x[:, 0]
        t1[:, 1, 1] = x[:, 1]
        t1[:, 2, 2] = x[:, 1]
        t2 = torch.zeros((x.size(0), 3, 3))
        t2[:, 0, 0] = x[:, 2]
        t2[:, 1, 1] = x[:, 3]
        t2[:, 2, 2] = x[:, 3]
        t3 = torch.zeros((x.size(0), 3, 3))
        t3[:, 0, 0] = x[:, 4]
        t3[:, 1, 1] = x[:, 5]
        t3[:, 2, 2] = x[:, 5]
        t4 = torch.zeros((x.size(0), 3, 3))
        t4[:, 0, 0] = x[:, 6]
        t4[:, 1, 1] = x[:, 7]
        t4[:, 2, 2] = x[:, 7]
        t5 = torch.zeros((x.size(0), 3, 3))
        t5[:, 0, 0] = x[:, 8]
        t5[:, 1, 1] = x[:, 9]
        t5[:, 2, 2] = x[:, 9]
        t6 = torch.zeros((x.size(0), 3, 3))
        t6[:, 0, 0] = x[:, 10]
        t6[:, 1, 1] = x[:, 11]
        t6[:, 2, 2] = x[:, 11]
        return t1, t2, t3, t4, t5, t6

    def _get_polarizability_vectors(self, x: Tensor) -> Tensor:
        """X should have size (_,3,3)."""
        indices = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]).T
        return x[:, indices[0], indices[1]]

    def _batch_graph(
        self, lattice: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate edge indexes, unit vectors, and changes in distance.

        Parameters
        ----------
        lattice
            Å | Tensor with size [S,3,3] where S is the number of samples.
        positions
            Unitless | Tensor with size [S,N,3] where N is the number of atoms.
        cutoff
            Å | Edge cutoff distance.

        Returns
        -------
        :
            3-tuple.
            First element is edge indexes, a tensor of size [3,X] where X is the number
            of edges. This tensor defines S non-interconnected graphs making up a
            batch. The first row defines the graph index. The second and third rows
            define the actual edge indexes used by ``triplet``. Second element is
            cartesian unit vectors, a tensor of size [X,3]. Third element is distances,
            a tensor of size [X,1].

        """
        num_samples = lattice.size(0)
        num_atoms = positions.size(1)

        # Get edge indexes
        graph_indexes = torch.tensor([range(num_samples)]).repeat_interleave(
            self._ref_edge_indexes.size(1), dim=1
        )
        edge_indexes = self._ref_edge_indexes.repeat((1, num_samples))
        edge_indexes[0] = graph_indexes

        # Compute pairwise distance matrix.
        displacement = positions.unsqueeze(1) - positions.unsqueeze(2)
        displacement = torch.where(
            displacement % 1 > 0.5, displacement % 1 - 1, displacement % 1
        )
        expanded_lattice = lattice[:, None, :, :].expand(
            -1, displacement.size(1), -1, -1
        )
        cart_displacement = displacement.matmul(expanded_lattice)
        cart_distance_matrix = torch.sqrt(torch.sum(cart_displacement**2, dim=-1))

        cart_unit_vectors = cart_displacement[
            edge_indexes[0], edge_indexes[1], edge_indexes[2]
        ]  # python 3.10 complains if we use the unpacking operator (*)
        cart_unit_vectors /= torch.linalg.norm(cart_unit_vectors, dim=-1)[:, None]
        distances = cart_distance_matrix[
            edge_indexes[0], edge_indexes[1], edge_indexes[2]
        ].view(-1, 1)
        # ref_distances = self._ref_distances.repeat((num_samples, 1))
        # distances -= ref_distances

        # Disconnect all S graphs
        edge_indexes[1] += edge_indexes[0] * num_atoms
        edge_indexes[2] += edge_indexes[0] * num_atoms

        return edge_indexes, cart_unit_vectors, distances

    def forward(  # pylint: disable=too-many-locals
        self,
        lattice: Tensor,
        atomic_numbers: Tensor,
        positions: Tensor,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        lattice
            Å | Tensor with size [S,3,3] where S is the number of samples.
        atomic_numbers
            Tensor with size [S,N] where N is the number of atoms.
        positions
            Unitless | Tensor with size [S,N,3].

        Returns
        -------
        :
            Polarizability tensors with size [S,6]. Polarizability is expressed in
            vector form. To convert into tensor form, see
            :func:`polarizability_vectors_to_tensors`.

        """
        edge_index, unit_vec, dist = self._batch_graph(lattice, positions)
        atom_types = self._convert_to_atom_type(atomic_numbers).flatten()

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = triplets(
            edge_index[[1, 2]], num_nodes=atom_types.size(0)
        )

        # Embedding blocks:
        node_emb = self._node_embedding(atom_types)
        edge_emb = self._edge_embedding(dist)

        # Message passing blocks:
        for node_block, edge_block in zip(self._node_blocks, self._edge_blocks):
            node_emb = node_block(node_emb, edge_emb, i)
            edge_emb = edge_block(
                node_emb, edge_emb, i, j, idx_i, idx_j, idx_k, idx_ji, idx_kj
            )

        # Polarizability prediction block:
        edge_polarizability = self._polarizability_predictor(edge_emb)

        t1, t2, t3, t4, t5, t6 = self._get_edge_polarizability_tensor(
            edge_polarizability
        )
        rotation = _get_rotations(unit_vec)

        t1 = rotation @ t1 @ torch.linalg.inv(rotation)
        t2 = rotation @ t2 @ torch.linalg.inv(rotation)
        t3 = rotation @ t3 @ torch.linalg.inv(rotation)
        t4 = rotation @ t4 @ torch.linalg.inv(rotation)
        t5 = rotation @ t5 @ torch.linalg.inv(rotation)
        t6 = rotation @ t6 @ torch.linalg.inv(rotation)

        mask_1 = torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        mask_2 = torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        mask_3 = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        mask_4 = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        mask_5 = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        mask_6 = torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])

        edge_polarizability = (
            t1 * mask_1
            + t2 * mask_2
            + t3 * mask_3
            + t4 * mask_4
            + t5 * mask_5
            + t6 * mask_6
        )
        edge_polarizability = self._get_polarizability_vectors(edge_polarizability)

        # Isolate polarizabilities from batch graphs.
        polarizability = torch.zeros((positions.size(0), 6))
        for i in range(positions.size(0)):
            mask = (edge_index[0:1] == i).T
            count = torch.sum(mask)
            polarizability[i] = torch.sum(edge_polarizability * mask, dim=0)
            polarizability[i] /= count
        return polarizability

        # component_polarizability = self.node_polarizability_predictor(node_emb)
        # colarizability = torch.zeros((positions.size(0), 6))
        # num_atoms = positions.size(1)
        # for i in range(positions.size(0)):
        #   mask = torch.zeros(1, atomic_numbers.size(0))
        #   mask[:, i * num_atoms : (i + 1) * num_atoms] = 1
        #   count = torch.sum(mask)
        #   polarizability[i] = torch.sum(component_polarizability * mask.T, dim=0)
        #   polarizability[i] /= count
        # return polarizability


def _radius_graph_pbc(
    lattice: Tensor, positions: Tensor, cutoff: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate graphs for structures while respecting periodic boundary conditions.

    Parameters
    ----------
    lattice
        Å | Tensor with size [S,3,3] where S is the number of samples.
    positions
        Unitless | Tensor with size [S,N,3] where N is the number of atoms.
    cutoff
        Å | Edge cutoff distance.

    Returns
    -------
    :
        3-tuple.
        First element is edge indexes, a tensor of size [3,X] where X is the number of
        edges. This tensor defines S non-interconnected graphs making up a batch. The
        first row defines the graph index. The second and third rows define the actual
        edge indexes used by ``triplet``.
        Second element is cartesian unit vectors, a tensor of size [X,3].
        Third element is distances, a tensor of side [X,1].

    """
    num_samples = lattice.size(0)
    num_atoms = positions.size(1)

    # Compute pairwise distance matrix.
    displacement = positions.unsqueeze(1) - positions.unsqueeze(2)
    displacement = torch.where(
        displacement % 1 > 0.5, displacement % 1 - 1, displacement % 1
    )
    expanded_lattice = lattice[:, None, :, :].expand(-1, displacement.size(1), -1, -1)
    cart_displacement = displacement.matmul(expanded_lattice)
    cart_distance_matrix = torch.sqrt(torch.sum(cart_displacement**2, dim=-1))

    # Compute adjacency matrix
    adjacency_matrix = cart_distance_matrix <= cutoff
    not_self_loop = ~torch.eye(adjacency_matrix.size(-1), dtype=torch.bool).expand(
        num_samples, -1, -1
    )
    adjacency_matrix = torch.logical_and(adjacency_matrix, not_self_loop)

    edge_indexes = torch.nonzero(adjacency_matrix).T
    cart_unit_vectors = cart_displacement[
        edge_indexes[0], edge_indexes[1], edge_indexes[2]
    ]
    cart_unit_vectors /= torch.linalg.norm(cart_unit_vectors, dim=-1)[:, None]
    distances = cart_distance_matrix[
        edge_indexes[0], edge_indexes[1], edge_indexes[2]
    ].view(-1, 1)

    # Disconnect all S graphs
    edge_indexes[1] += edge_indexes[0] * num_atoms
    edge_indexes[2] += edge_indexes[0] * num_atoms

    return edge_indexes, cart_unit_vectors, distances
