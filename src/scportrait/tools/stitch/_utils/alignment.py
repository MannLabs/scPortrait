from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import numpy as np
from typing import Tuple, Union

def apply_shift(target_img: np.ndarray, shift: np.ndarray) -> np.ndarray:
    """Applies a Fourier shift to the target image.

    This function shifts the input image in the Fourier domain by the specified amount
    and returns the corrected image. This image is usually another staining from another cycle.

    Args:
        target_img: The input image to be shifted.
        shift (np.ndarray): The amount of shift to apply.

    Returns:
        corrected_image (np.ndarray): The shifted image.
    """
    corrected_image = fourier_shift(np.fft.fftn(target_img), np.array(shift))
    corrected_image = np.fft.ifftn(corrected_image).real  # Take real part
    return corrected_image


def get_registered_img(
    source_img: np.ndarray, 
    target_img: np.ndarray, 
    normalize_images: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """Registers the target image to the source image using phase cross-correlation.

    This function calculates the shift required to align the target image with the source image 
    using phase cross-correlation and applies that shift. The shift can also be used for 
    aligning other stainings of the same cycle as the target image.

    Based on:
    https://scikit-image.org/docs/0.23.x/auto_examples/registration/plot_register_translation.html

    Args:
        source_img (np.ndarray): The reference image to which the target image will be aligned. 
            Usually the DAPI staining of a cycle.
        target_img (np.ndarray): The image to be registered to the source image. 
            Usually the DAPI staining of another cycle.
        normalize_images (bool, optional): If True, normalize images to the range [0, 1] before registration. 
            Defaults to True.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing:
            corrected_image (np.ndarray): The registered (shifted) target image.
            shift (np.ndarray): The shift vector that was applied to align the target image.
    """
    if normalize_images:
        source_img = (source_img - source_img.min()) / (source_img.max() - source_img.min())
        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())
    shift, error, diffphase = phase_cross_correlation(source_img, target_img, upsample_factor=100)
    corrected_image, _ = apply_shift(target_img, shift)  # Apply shift
    return corrected_image, shift