"""Polarizability model based on a graph neural network (GNN)."""

from __future__ import annotations

import typing

from ramannoodle.structure.reference import ReferenceStructure
from ramannoodle.exceptions import get_torch_missing_error, UsageError

try:
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
    from torch_geometric.nn.inits import reset
    from torch_geometric.nn.models.schnet import ShiftedSoftplus
    from torch_geometric.utils import scatter
    import ramannoodle.polarizability.torch.utils as rn_torch_utils
except (ModuleNotFoundError, UsageError) as exc:
    raise get_torch_missing_error() from exc

# pylint: disable=not-callable


class GaussianFilter(torch.nn.Module):
    """Gaussian filter.

    Parameters should be chosen such that all expected inputs are between lower and
    upper bounds.

    Parameters
    ----------
    lower_bound
    upper_bound
    steps
        | Number of steps to take between lower_bound and upper_bound.
    """

    def __init__(self, lower_bound: float, upper_bound: float, steps: int):
        super().__init__()
        offset = torch.linspace(lower_bound, upper_bound, steps)
        self.coefficient = -0.5 / (float(offset[1]) - float(offset[0])) ** 2
        self.register_buffer("offset", offset)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x
            | 1D tensor with size [D,]. Typically contains interatomic distances.

        Returns
        -------
        :
            2D tensor with size [D,steps].

        """
        x = x.view(-1, 1) - self.offset.view(1, -1)
        return torch.exp(self.coefficient * x.pow(2))


class NodeBlock(torch.nn.Module):
    """Edge to node message passer.

    Architecture and notation is based on equation (5) in https://doi.org/10.1038/
    s41524-021-00543-3. The architecture has been modified to use layer normalization.

    Parameters
    ----------
    size_node_embedding
    size_edge_embedding

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
        self.c1_linear = Linear(
            size_node_embedding + size_edge_embedding,
            2 * size_node_embedding,
        )

        self.c1_norm = LayerNorm(2 * size_node_embedding)
        self.final_norm = LayerNorm(size_node_embedding)

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        self.c1_linear.reset_parameters()
        self.c1_norm.reset_parameters()
        self.final_norm.reset_parameters()

    def forward(
        self, node_embedding: Tensor, edge_embedding: Tensor, i: Tensor
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        node_embedding
            | 2D tensor with size [N,size_node_embedding] where N is the number of
            | nodes.
        edge_embedding
            | 2D tensor with size [E,size_edge_embedding] where E is the number of
            | edges.

        Returns
        -------
        :
            2D tensor with size [N,size_node_embedding].
        """
        c1 = torch.cat([node_embedding[i], edge_embedding], dim=1)
        c1 = self.c1_norm(self.c1_linear(c1))
        c1_filter, c1_core = c1.chunk(2, dim=1)
        c1_filter = c1_filter.sigmoid()
        c1_core = c1_core.tanh()
        c1_emb = scatter(
            c1_filter * c1_core, i, dim=0, dim_size=node_embedding.size(0), reduce="sum"
        )
        c1_emb = self.final_norm(c1_emb)

        return typing.cast(Tensor, (node_embedding + c1_emb).tanh())


class EdgeBlock(torch.nn.Module):
    """Node to edge message passer.

    Architecture and notation is based on equation (6) in https://doi.org/10.1038/
    s41524-021-00543-3. The architecture has been modified to use batch normalization.

    Parameters
    ----------
    size_node_embedding
    size_edge_embedding
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
        self.c2_linear = Linear(size_node_embedding, 2 * size_edge_embedding)

        # Combination of two linear layers, dubbed "core" and "filter":
        #   "filter" : c3ij W5 + b5
        #   "core"   : c3ij W6 + b6
        self.c3_linear = Linear(
            3 * size_node_embedding + 2 * size_edge_embedding,
            2 * size_edge_embedding,
        )

        self.c2_norm_1 = LayerNorm(2 * size_edge_embedding)
        self.c3_norm_1 = LayerNorm(2 * size_edge_embedding)
        self.c2_norm_2 = LayerNorm(size_edge_embedding)
        self.c3_norm_2 = LayerNorm(size_edge_embedding)

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        self.c2_linear.reset_parameters()
        self.c3_linear.reset_parameters()
        self.c2_norm_1.reset_parameters()
        self.c3_norm_1.reset_parameters()
        self.c2_norm_2.reset_parameters()
        self.c3_norm_2.reset_parameters()

    def _get_c2_embedding(
        self,
        node_embedding: Tensor,
        i: Tensor,
        j: Tensor,
    ) -> Tensor:
        """Get c2 embedding.

        Parameters
        ----------
        node_embedding
            | 2D tensor with size [N,size_node_embedding] where N is the number of
            | nodes.
        i
            | Node 1 of edge pairs, a 1D tensor with size [E,].
        j
            | Node 2 of edge pairs, a 1D tensor with size [E,].

        Returns
        -------
        :
            2D tensor with size [E,size_edge_embedding].
        """
        c2 = node_embedding[i] * node_embedding[j]
        c2 = self.c2_norm_1(self.c2_linear(c2))
        c2_filter, c2_core = c2.chunk(2, dim=1)
        c2_filter = c2_filter.sigmoid()
        c2_core = c2_core.tanh()
        return typing.cast(Tensor, self.c2_norm_2(c2_filter * c2_core))

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
        """Get c3 embedding.

        Parameters
        ----------
        node_embedding
            | 2D tensor with size [N,size_node_embedding] where N is the number of
            | nodes.
        edge_embedding
            | 2D tensor with size [E,size_edge_embedding] where E is the number of
            | edges.
        index_i
            | Node 1 of edge triplets, a 1D tensor with size [T,] where T is the number
            | of triplets.
        index_j
            | Node 2 of edge triplets, a 1D tensor with size [T,].
        index_k
            | Node 3 of edge triplets, a 1D tensor with size [T,].
        index_ji
            | Index of (j,i) corresponding to (index_j,index_i), a 1D tensor with size
            | [T,.]
        index_kj
            | Index of (k,j) corresponding to (index_k,index_j), a 1D tensor with size
            | [T,.]

        Returns
        -------
        :
            2D tensor with size [E,size_edge_embedding].
        """
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
        c3 = self.c3_norm_1(self.c3_linear(c3))
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
        return typing.cast(Tensor, self.c3_norm_2(c3_emb))

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
        """Forward pass.

        Parameters
        ----------
        node_embedding
            | 2D tensor with size [N,size_node_embedding] where N is the number of
            | nodes.
        edge_embedding
            | 2D tensor with size [E,size_edge_embedding] where E is the number of
            | edges.
        i
            | Node 1 of edge pairs, a 1D tensor with size [E,].
        j
            | Node 2 of edge pairs, a 1D tensor with size [E,].
        index_i
            | Node 1 of edge triplets, a 1D tensor with size [T,] where T is the number
            | of triplets.
        index_j
            | Node 2 of edge triplets, a 1D tensor with size [T,].
        index_k
            | Node 3 of edge triplets, a 1D tensor with size [T,].
        index_ji
            | Index of (j,i) corresponding to (index_j,index_i), a 1D tensor with size
            | [T,.]
        index_kj
            | Index of (k,j) corresponding to (index_k,index_j), a 1D tensor with size
            | [T,.]

        Returns
        -------
        :
            2D tensor with size [E,size_edge_embedding].

        """
        c2_embedding = self._get_c2_embedding(node_embedding, i, j)
        # pylint: disable=duplicate-code
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


def _get_edge_polarizability_vectors(
    polarizability_embedding: Tensor,
    unit_vector: Tensor,
) -> Tensor:
    """Convert polarizability embedding to edge polarizability vectors.

    Parameters
    ----------
    polarizability_embedding
        | 2D tensor with size [E,12] where E is the number of edges.
    unit_vector
        | (Å) Unit vectors of edges, a 2D tensor with size [E,3].

    Returns
    -------
    :
        2D tensor with size [E,6].
    """
    a1 = torch.zeros((polarizability_embedding.size(0), 3, 3))
    a1[:, 0, 0] = polarizability_embedding[:, 0]
    a1[:, 1, 1] = polarizability_embedding[:, 1]
    a1[:, 2, 2] = polarizability_embedding[:, 1]
    a2 = torch.zeros((polarizability_embedding.size(0), 3, 3))
    a2[:, 0, 0] = polarizability_embedding[:, 2]
    a2[:, 1, 1] = polarizability_embedding[:, 3]
    a2[:, 2, 2] = polarizability_embedding[:, 3]
    a3 = torch.zeros((polarizability_embedding.size(0), 3, 3))
    a3[:, 0, 0] = polarizability_embedding[:, 4]
    a3[:, 1, 1] = polarizability_embedding[:, 5]
    a3[:, 2, 2] = polarizability_embedding[:, 5]
    a4 = torch.zeros((polarizability_embedding.size(0), 3, 3))
    a4[:, 0, 0] = polarizability_embedding[:, 6]
    a4[:, 1, 1] = polarizability_embedding[:, 7]
    a4[:, 2, 2] = polarizability_embedding[:, 7]
    a5 = torch.zeros((polarizability_embedding.size(0), 3, 3))
    a5[:, 0, 0] = polarizability_embedding[:, 8]
    a5[:, 1, 1] = polarizability_embedding[:, 9]
    a5[:, 2, 2] = polarizability_embedding[:, 9]
    a6 = torch.zeros((polarizability_embedding.size(0), 3, 3))
    a6[:, 0, 0] = polarizability_embedding[:, 10]
    a6[:, 1, 1] = polarizability_embedding[:, 11]
    a6[:, 2, 2] = polarizability_embedding[:, 11]

    rotation = rn_torch_utils.get_rotations(unit_vector)
    inv_rotation = torch.linalg.inv(rotation)

    a1 = rotation @ a1 @ inv_rotation
    a2 = rotation @ a2 @ inv_rotation
    a3 = rotation @ a3 @ inv_rotation
    a4 = rotation @ a4 @ inv_rotation
    a5 = rotation @ a5 @ inv_rotation
    a6 = rotation @ a6 @ inv_rotation

    edge_polarizability = (
        a1 * torch.tensor([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        + a2 * torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        + a3 * torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        + a4 * torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        + a5 * torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        + a6 * torch.tensor([[0, 0, 0], [0, 0, 0], [0, 0, 1]])
    )
    return rn_torch_utils.get_polarizability_vectors(edge_polarizability)


class PotGNN(Module):  # pylint: disable=too-many-instance-attributes
    r"""POlarizability Tensor Graph Neural Network (PotGNN).

    GNN architecture was inspired by the "direct force architecture" developed in Park
    `et al.`;  `npj Computational Materials` (2021)7:73; https://doi.org/10.1038/
    s41524-021-00543-3. Implementation adapted from ``torch_geometric.nn.models.GNNFF``
    authored by @ken2403 and merged by @rusty1s.

    Parameters
    ----------
    ref_structure
        | Reference structure from which nodes/edges are determined.
    cutoff
        | (Å) Cutoff distance for edges.
    size_node_embedding
    size_edge_embedding
    num_message_passes
    gaussian_filter_start
        | (Å) Lower bound of the Gaussian filter used in initial edge embedding.
    gaussian_filter_end
        | (Å) Upper bound of the Gaussian filter used in initial edge embedding.
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-instance-attributes
        self,
        ref_structure: ReferenceStructure,
        cutoff: float,
        size_node_embedding: int,
        size_edge_embedding: int,
        num_message_passes: int,
        gaussian_filter_start: float,
        gaussian_filter_end: float,
    ):
        super().__init__()

        self._ref_structure = ref_structure
        self._cutoff = cutoff

        # Set up graph.
        lattice = torch.from_numpy(ref_structure.lattice).unsqueeze(0)
        positions = torch.from_numpy(ref_structure.positions).unsqueeze(0)
        self._ref_edge_indexes, _, _ = rn_torch_utils._radius_graph_pbc(
            lattice.type(torch.get_default_dtype()).to(torch.get_default_device()),
            positions.type(torch.get_default_dtype()).to(torch.get_default_device()),
            cutoff,
        )
        self._batch_triplets = rn_torch_utils.BatchTriplets(
            len(ref_structure.atomic_numbers), self._ref_edge_indexes
        )

        # Atom types map
        unique_atomic_numbers = set(ref_structure.atomic_numbers)
        self._num_atom_types = len(unique_atomic_numbers)
        self._atom_type_map = (torch.zeros(119) - 1).type(torch.int)
        for atom_type, atomic_number in enumerate(unique_atomic_numbers):
            self._atom_type_map[atomic_number] = atom_type

        self._node_embedding = Sequential(
            Embedding(self._num_atom_types, size_node_embedding),
            ShiftedSoftplus(),
            Linear(size_node_embedding, size_node_embedding),
            ShiftedSoftplus(),
            Linear(size_node_embedding, size_node_embedding),
        )
        self._edge_embedding = GaussianFilter(
            gaussian_filter_start, gaussian_filter_end, size_edge_embedding
        )

        self._node_blocks = ModuleList(
            [
                NodeBlock(size_node_embedding, size_edge_embedding)
                for _ in range(num_message_passes)
            ]
        )
        self._edge_blocks = ModuleList(
            [
                EdgeBlock(size_node_embedding, size_edge_embedding)
                for _ in range(num_message_passes)
            ]
        )

        self._to_polarizability_embedding = Sequential(
            Linear(size_edge_embedding, size_edge_embedding),
            BatchNorm1d(size_edge_embedding),
            ShiftedSoftplus(),
            Linear(size_edge_embedding, size_edge_embedding),
            ShiftedSoftplus(),
            Linear(size_edge_embedding, 12),
        )

    def _convert_to_atom_type(self, atomic_numbers: Tensor) -> Tensor:
        """Convert atomic numbers into atom types.

        Atom types must be defined for all atomic numbers.

        Parameters
        ----------
        atomic_numbers
            | Tensor with arbitrary shape.

        Returns
        -------
        :
            Tensor with the same shape as atomic_numbers.

        """
        return self._atom_type_map[atomic_numbers]

    def reset_parameters(self) -> None:
        """Reset model parameters."""
        reset(self._node_embedding)
        self._edge_embedding.reset_parameters()
        for node_block in self._node_blocks:
            node_block.reset_parameters()
        for edge_block in self._edge_blocks:
            edge_block.reset_parameters()
        reset(self._to_polarizability_embedding)

    def _batch_graph(
        self, lattice: Tensor, positions: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Generate edge indexes, unit vectors, and edge distances for a batch.

        Parameters
        ----------
        lattice
            | (Å) Tensor with size [S,3,3] where S is the number of samples.
        positions
            | (fractional) Tensor with size [S,N,3] where N is the number of atoms.

        Returns
        -------
        :
            3-tuple:
            0. | edge indexes --
               | 2D Tensor of size [3,E] where E is the number of edges. The first
               | element are the graph indexes, while the remaining two elements are
               | edge indexes.
            #. | unit vectors --
               | (Å) 2D Tensor with size [E,3].
            #. | distances --
               | (Å) 1D Tensor with size [E,].

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

        return rn_torch_utils.get_graph_info(
            cart_displacement, edge_indexes, cart_distance_matrix, num_atoms
        )

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
            | (Å) 3D tensor with size [S,3,3] where S is the number of samples.
        atomic_numbers
            | Tensor with size [S,N] where N is the number of atoms.
        positions
            | (fractional) Tensor with size [S,N,3].

        Returns
        -------
        :
            Polarizability vectors with size [S,6]. To convert into tensor form, see
            :func:`polarizability_vectors_to_tensors`.

        """
        edge_index, unit_vector, distance = self._batch_graph(lattice, positions)
        atom_types = self._convert_to_atom_type(atomic_numbers).flatten()
        node_emb = self._node_embedding(atom_types)
        edge_emb = self._edge_embedding(distance)

        # Message passing blocks:
        triplets = self._batch_triplets.get_triplets(batch_size=lattice.size(0))
        for node_block, edge_block in zip(self._node_blocks, self._edge_blocks):
            node_emb = node_block(node_emb, edge_emb, triplets[0])
            edge_emb = edge_block(node_emb, edge_emb, *triplets)

        # Get edge polarizabilities.
        polarizability_emb = self._to_polarizability_embedding(edge_emb)
        edge_polarizability = _get_edge_polarizability_vectors(
            polarizability_emb, unit_vector
        )

        # Isolate polarizabilities from batch graphs.
        polarizability = torch.zeros((positions.size(0), 6))
        for structure_i in range(positions.size(0)):
            mask = (edge_index[0:1] == structure_i).T
            count = torch.sum(mask)
            polarizability[structure_i] = torch.sum(edge_polarizability * mask, dim=0)
            polarizability[structure_i] /= count
        return polarizability
