from . import write
from . import processing
from . import processing as pp

# has to be done at the end, after everything has been imported
import sys
sys.modules.update({f"{__name__}.{m}": globals()[m] for m in ["pp"]})

