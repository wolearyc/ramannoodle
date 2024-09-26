"""Training functions for PotGNN polarizability model."""

import numpy as np
from numpy.typing import NDArray

from ramannoodle.exceptions import (  # pylint: disable=ungrouped-imports
    get_torch_missing_error,
    UserError,
)

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.nn.modules.loss import _Loss
    from torch.optim.optimizer import Optimizer

    from ramannoodle.pmodel.torch.gnn import PotGNN
    from ramannoodle.dataset.torch.dataset import PolarizabilityDataset
except (ModuleNotFoundError, UserError) as exc:
    raise get_torch_missing_error() from exc


# pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
def train_single_epoch(
    model: PotGNN,
    training_set: PolarizabilityDataset,
    validation_set: PolarizabilityDataset,
    batch_size: int,
    optimizer: Optimizer,
    loss_function: _Loss,
) -> tuple[float, float, NDArray[np.float64]]:
    """Train PotGNN model for a single epoch on the default device.

    Parameters
    ----------
    model
    training_set
    validation_set
    batch_size
    optimizer
    loss_function

    Returns
    -------
    :
        0.  mean training loss
        #.  mean validation loss
        #.  mean variance of predictions on validation set -- Array with shape [6,]

    """
    default_device = torch.get_default_device()

    train_loader = DataLoader(
        training_set,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator(device=default_device),
    )
    validation_loader = DataLoader(
        validation_set, batch_size=min(100, len(validation_set)), shuffle=False
    )

    model.train()
    train_losses = []
    for lattice, atomic_numbers, position, polarizability in train_loader:
        out = model.forward(
            lattice.to(default_device),
            atomic_numbers.to(default_device),
            position.to(default_device),
        )
        loss = loss_function(out, polarizability)
        train_losses.append(float(loss))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    model.eval()
    validation_losses = []
    validation_vars = []
    for lattice, atomic_numbers, position, polarizability in validation_loader:
        out = model.forward(
            lattice.to(default_device),
            atomic_numbers.to(default_device),
            position.to(default_device),
        )
        loss = loss_function(out, polarizability)
        validation_losses.append(float(loss))
        validation_vars.append(torch.var(out, dim=0).detach().cpu().numpy().copy())

    return (
        float(np.mean(train_losses)),
        float(np.mean(validation_losses)),
        np.mean(validation_vars, axis=0),
    )
