"""Top-level package for scPortrait"""

__version__ = "1.3.1-dev0"

# silence warnings
import warnings

from scportrait import io
from scportrait import pipeline as pipeline
from scportrait import plotting as pl
from scportrait import processing as pp
from scportrait import tools as tl

# Python 3.12 is more strict about escape sequencing in string literals
# mahotas: https://github.com/luispedro/mahotas/issues/151
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

# silence warning from spatialdata resulting in an older dask version see #139
warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")

# silence warning from cellpose resulting in missing parameter set in model call see #141
warnings.filterwarnings(
    "ignore", message=r"You are using `torch.load` with `weights_only=False`.*", category=FutureWarning
)


# silence warning
warnings.filterwarnings("ignore", category=FutureWarning, message="The plugin infrastructure in")
