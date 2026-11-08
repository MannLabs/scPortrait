"""Tile blending for mosaic assembly.

Drop-in replacement for ``ashlar.utils.pastefunc_blend`` that avoids the deprecated
scikit-image APIs used by ashlar (``remove_small_holes(area_threshold=...)`` and
``skimage.morphology.binary_dilation``, which is removed in scikit-image 0.28).
The output is identical to ashlar's original implementation.
"""

import inspect
import math

import numpy as np
import scipy.ndimage
import skimage.morphology

_HAS_MAX_SIZE = "max_size" in inspect.signature(skimage.morphology.remove_small_holes).parameters


def _remove_small_holes(mask: np.ndarray, area_threshold: float) -> np.ndarray:
    """Remove holes strictly smaller than ``area_threshold`` (pre-0.26 semantics) on any scikit-image version."""
    if _HAS_MAX_SIZE:
        # max_size removes holes <= max_size, area_threshold removed holes < area_threshold
        return skimage.morphology.remove_small_holes(mask, max_size=math.ceil(area_threshold) - 1)
    return skimage.morphology.remove_small_holes(mask, area_threshold=area_threshold)


def pastefunc_blend(target: np.ndarray, img: np.ndarray) -> np.ndarray:
    """Linear blend based on distance to unfilled space in target (see ``ashlar.utils.pastefunc_blend``)."""
    hole_threshold = np.mean(target.shape)
    mask = _remove_small_holes(target != 0, hole_threshold)
    dist = scipy.ndimage.distance_transform_cdt(mask)
    dmax = dist.max()
    if dmax == 0:
        alpha = 0
    else:
        alpha = dist / dmax
        # keep target pixel values where img is 0 (1-pixel dilation to clean up the edge);
        # scipy's default structuring element equals skimage's default footprint here
        alpha[scipy.ndimage.binary_dilation(img == 0)] = 1
    return target * alpha + img * (1 - alpha)
