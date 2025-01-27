"""Top-level package for scPortrait"""

# silence warnings
import warnings

from scportrait import io
from scportrait import pipeline as pipeline
from scportrait import plotting as pl
from scportrait import processing as pp
from scportrait import tools as tl

# silence warning from spatialdata resulting in an older dask version see #139
warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")
