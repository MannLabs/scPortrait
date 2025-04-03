"""
spdata
=======

Functions to work with spatialdata objects.
"""

# has to be done at the end, after everything has been imported
import sys

from . import processing as pp
from .write._helper import add_element_sdata, rename_image_element
from .write._write import image as write_image

# update symlinks to the functions
sys.modules.update(
    {
        f"{__name__}.{m}": globals()[m]
        for m in [
            "pp",
            "write_image",
        ]
    }
)

__all__ = ["write_image", "add_element_sdata", "rename_image_element" "pp"]
