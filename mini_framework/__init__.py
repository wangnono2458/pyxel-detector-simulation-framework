"""Minimal simulation framework inspired by Pyxel."""

# Register built-in models (side-effect registration)
from mini_framework.models import (  # noqa: F401
    scene,
    photon,
    charge,
    measurement,
    readout,
    data_processing,
)
