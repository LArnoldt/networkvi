"""scvi-tools."""

# Set default logging handler to avoid logging with logging.lastResort logger.
import logging
import warnings

from ._constants import REGISTRY_KEYS
from ._settings import settings

# this import needs to come after prior imports to prevent circular import
from . import data, model, utils

from importlib.metadata import version

package_name = "networkvi"
__version__ = "1.0.0"

settings.verbosity = logging.INFO

# Jax sets the root logger, this prevents double output.
scvi_logger = logging.getLogger("networkvi")
scvi_logger.propagate = False


__all__ = [
    "settings",
    "REGISTRY_KEYS",
    "data",
    "model",
    "utils",
]

