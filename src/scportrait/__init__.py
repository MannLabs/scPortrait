"""Top-level package for scPortrait"""

__version__ = "1.6.1-dev0"

import sys
import warnings

# Python 3.12 is more strict about escape sequencing in string literals
# mahotas: https://github.com/luispedro/mahotas/issues/151
warnings.filterwarnings("ignore", category=SyntaxWarning, message="invalid escape sequence")

# silence warning from spatialdata resulting in an older dask version see #139
warnings.filterwarnings("ignore", message="ignoring keyword argument 'read_only'")
warnings.filterwarnings("ignore", message=".*legacy Dask DataFrame implementation is deprecated and will be removed")

# silence warning from cellpose resulting in missing parameter set in model call see #141
warnings.filterwarnings(
    "ignore", message=r"You are using `torch.load` with `weights_only=False`.*", category=FutureWarning
)

# silence warning
warnings.filterwarnings("ignore", category=FutureWarning, message="The plugin infrastructure in")

from . import data, io, pipeline
from . import plotting as pl
from . import processing as pp
from . import tools as tl

# has to be done at the end, after everything has been imported
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["tl", "pp", "pl"]})
