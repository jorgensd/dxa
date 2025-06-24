"""Top-level package for dxa."""

from .types import Function
from .function import assign
from .assembly import assemble_scalar
from importlib.metadata import metadata

meta = metadata("dolfinx_adjoint")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]


__all__ = [
    "Function",
    "assemble_scalar",
    "assign",
    "__version__",
    "__author__",
    "__license__",
    "__email__",
    "__program_name__",
]
