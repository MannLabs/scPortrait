"""
stitch
=======

Functions to assemble tiled images into fullscale mosaics.
Uses out-of-memory computation for the assembly of larger than memory image mosaics.
"""

from ._stitch import ParallelStitcher, Stitcher

__all__ = ["Stitcher", "ParallelStitcher"]
