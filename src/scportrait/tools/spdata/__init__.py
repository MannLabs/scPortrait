# has to be done at the end, after everything has been imported
import sys

from . import processing, write
from . import processing as pp

sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp"]})
