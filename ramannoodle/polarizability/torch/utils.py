"""Utility functions for PyTorch models."""

from typing import Sequence

from ramannoodle.exceptions import get_type_error, get_torch_missing_error

try:
    import torch
    from torch import Tensor
    from torch_geometric.nn.models.dimenet import triplets as dimenet_triplets
except ModuleNotFoundError as exc:
    raise get_torch_missing_error() from exc

# pylint complains about torch.norm
# pylint: disable=not-callable


def polarizability_vectors_to_tensors(polarizability_vectors: Tensor) -> Tensor:
    """Convert polarizability vectors to symmetric tensors.

    Parameters
    ----------
    polarizability_vectors
        | 2D Tensor with size [S,6].

    Returns
    -------
    :
        3D tensor with size [S,3,3].
    """
    verify_tensor_size("polarizability_vectors", polarizability_vectors, (None, 6))
    indices = torch.tensor(
        [
            [0, 3, 4],
            [3, 1, 5],
            [4, 5, 2],
        ]
    )
    return polarizability_vectors[:, indices]


def polarizability_tensors_to_vectors(polarizability_tensors: Tensor) -> Tensor:
    """Convert polarizability tensors to vectors.

    Parameters
    ----------
    polarizability_tensors
        | 3D tensor with size [S,3,3] where S is the number of samples.

    Returns
    -------
    :
        2D tensor with size [S,6].

    """
    verify_tensor_size("polarizability_tensors", polarizability_tensors, (None, 3, 3))
    indices = torch.tensor([[0, 0], [1, 1], [2, 2], [0, 1], [0, 2], [1, 2]]).T
    return polarizability_tensors[:, indices[0], indices[1]]


def _get_tensor_size_str(size: Sequence[int | None]) -> str:
    """Get a string representing a tensor size.

    "_" indicates a dimension can be any size.

    Parameters
    ----------
    size
        | None indicates dimension can be any size.
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


def get_tensor_size_error(name: str, tensor: Tensor, desired_size: str) -> ValueError:
    """Get ValueError indicating a PyTorch Tensor has the wrong size."""
    try:
        shape_spec = f"{_get_tensor_size_str(tensor.size())} != {desired_size}"
    except AttributeError as exc:
        raise get_type_error("tensor", tensor, "Tensor") from exc
    return ValueError(f"{name} has wrong size: {shape_spec}")


def verify_tensor_size(name: str, tensor: Tensor, size: Sequence[int | None]) -> None:
    """Verify a PyTorch Tensor's size.

    :meta private: We should avoid calling this function whenever possible (EATF).

    Parameters
    ----------
    size
        int elements will be checked, None elements will not be.
    """
    try:
        if len(size) != tensor.ndim:
            raise get_tensor_size_error(name, tensor, _get_tensor_size_str(size))
        for d1, d2 in zip(tensor.size(), size, strict=True):
            if d2 is not None and d1 != d2:
                raise get_tensor_size_error(name, tensor, _get_tensor_size_str(size))
    except AttributeError as exc:
        raise get_type_error(name, tensor, "Tensor") from exc


def get_rotations(targets: Tensor) -> Tensor:
    """Get rotation matrices from (1,0,0) to target vectors.

    Parameters
    ----------
    targets
        | 2D tensor with size [S,3]. Vectors do not need to be normalized.

    Returns
    -------
    :
        3D tensor with size [S,3,3].
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

    # a==b implies s==0, which implies rotation should be identity/
    rotations[s == 0] = torch.eye(3)

    return rotations


def get_graph_info(
    cart_displacement: Tensor,
    edge_indexes: Tensor,
    cart_distance_matrix: Tensor,
    num_atoms: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Get information on graph."""
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


def _radius_graph_pbc(
    lattice: Tensor, positions: Tensor, cutoff: float
) -> tuple[Tensor, Tensor, Tensor]:
    """Generate graph for structures while respecting periodic boundary conditions.

    Parameters
    ----------
    lattice
        | (Å) 3D tensor with size [S,3,3] where S is the number of samples.
    positions
        | (fractional) 3D tensor with size [S,N,3] where N is the number of atoms.
    cutoff
        | Edge cutoff distance.

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

    return get_graph_info(
        cart_displacement, edge_indexes, cart_distance_matrix, num_atoms
    )


class BatchTriplets:
    """Graph triplets.

    Parameters
    ----------
    num_nodes
    ref_edge_indexes
    """

    def __init__(self, num_nodes: int, ref_edge_indexes: Tensor):
        self._num_nodes = num_nodes
        self._num_edges = len(ref_edge_indexes[1])
        with torch.device("cpu"):
            self._ref_triplets = dimenet_triplets(
                edge_index=ref_edge_indexes[[1, 2]].to("cpu"),
                num_nodes=self._num_nodes,
            )
        self._ref_triplets = tuple(
            t.to(torch.get_default_device()) for t in self._ref_triplets
        )
        assert len(self._ref_triplets) == 7  # type hint
        self._num_triplets = self._ref_triplets[2].size(0)

        self._cached_batch_size = 1
        self._cached_triplets = self._ref_triplets

    @property
    def cached_batch_size(self) -> int:
        """Get batch size of current cache."""
        return self._cached_batch_size

    def get_triplets(
        self, batch_size: int
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Get triplets for a given batch size.

        Returns
        -------
        :
            7-tuple:
                0. | i --
                   | Node 1 of edge pairs, a 1D tensor with size [E,].
                #. | j --
                   | Node 2 of edge pairs, a 1D tensor with size [E,].
                #. | index_i --
                   | Node 1 of edge triplets, a 1D tensor with size [T,] where T is the
                   | number of triplets.
                #. | index_j --
                   | Node 2 of edge triplets, a 1D tensor with size [T,].
                #. | index_k --
                   | Node 3 of edge triplets, a 1D tensor with size [T,].
                #. | index_ji --
                   | Index of (j,i) corresponding to (index_j,index_i), a 1D tensor
                   | with size [T,].
                #. | index_kj --
                   | Index of (k,j) corresponding to (index_k,index_j), a 1D tensor
                   | with size [T,].

        """
        if batch_size != self._cached_batch_size:
            i, j, index_i, index_j, index_k, index_ji, index_kj = tuple(
                t.repeat(batch_size) for t in self._ref_triplets
            )

            batch_indexes = torch.tensor([range(batch_size)]).repeat_interleave(
                self._num_edges, dim=1
            )[0]
            for index in [i, j]:
                index += batch_indexes * self._num_nodes

            batch_indexes = torch.tensor([range(batch_size)]).repeat_interleave(
                self._num_triplets, dim=1
            )[0]
            for index in [index_i, index_j, index_k]:
                index += batch_indexes * self._num_nodes
            for index in [index_ji, index_kj]:
                index += batch_indexes * self._num_edges

            self._cached_triplets = (
                i,
                j,
                index_i,
                index_j,
                index_k,
                index_ji,
                index_kj,
            )
            self._cached_batch_size = batch_size

        return self._cached_triplets
