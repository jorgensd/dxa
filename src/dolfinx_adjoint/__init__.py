"""Top-level package for dxa."""
from importlib.metadata import metadata

meta = metadata("dolfinx_adjoint")
__version__ = meta["Version"]
__author__ = meta["Author"]
__license__ = meta["License"]
__email__ = meta["Author-email"]
__program_name__ = meta["Name"]
