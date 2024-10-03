"""Classes representing atomic motions, for example phonons and trajectories."""

# flake8: noqa: F401
from ramannoodle.dynamics._phonon import Phonons
from ramannoodle.dynamics._trajectory import Trajectory

__all__ = ["Phonons", "Trajectory"]
