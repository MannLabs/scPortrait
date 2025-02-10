"""Top-level package for scPortrait"""

__version__ = "2.0.0"

# silence warnings
import warnings

from scportrait import io
from scportrait import pipeline as pipeline
from scportrait import plotting as pl
from scportrait import processing as pp
from scportrait import tools as tl

# silence warning from spatialdata resulting in an older dask version see #139
warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")

# silence warning from cellpose resulting in missing parameter set in model call see #141
warnings.filterwarnings(
    "ignore", message=r"You are using `torch.load` with `weights_only=False`.*", category=FutureWarning
)
