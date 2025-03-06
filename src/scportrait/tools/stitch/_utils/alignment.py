from skimage.registration import phase_cross_correlation
from scipy.ndimage import fourier_shift
import numpy as np

def apply_shift(target_img, shift):
    """
    Apply a Fourier shift to the target image.

    This function shifts the input image in the Fourier domain by the specified amount
    and returns the corrected image. This image is usually another staining from another cycle.

    Parameters:
    target_img (numpy.ndarray): The input image to be shifted.
    shift (array-like): The amount of shift to apply to the image. This should be an array-like
                        object with the same number of dimensions as the target image.

    Returns:
    tuple: A tuple containing:
        - corrected_image (numpy.ndarray): The shifted image.
        - shift (array-like): The applied shift.
    """
    corrected_image = fourier_shift(np.fft.fftn(target_img), np.array(shift))
    corrected_image = np.fft.ifftn(corrected_image).real  # Take real part
    return corrected_image


def get_registered_img(source_img, target_img, normalize_images=True):
    """
    Registers the target image to the source image using phase cross-correlation. This function is based on
    https://scikit-image.org/docs/0.23.x/auto_examples/registration/plot_register_translation.html#sphx-glr-auto-examples-registration-plot-register-translation-py.

    The function expects 2 grayscale images of the same size. The target image will be registered to the source image
    by applying a shift. The shift is calculated using phase cross-correlation. Shift is returned to be applied on other stainings of the same cycle of the target_img.
    Parameters:
    source_img (numpy.ndarray): The source image to which the target image will be registered. Usually the DAPI staining of a cycle.
    target_img (numpy.ndarray): The target image that will be registered to the source image. DAPI staining of another cycle.
    normalize_images (bool): If True, normalize the images to the range [0, 1] before registration.

    Returns:
    numpy.ndarray: The registered (corrected) target image.
    numpy.ndarray: The shift vector that was applied to the target image.
    """
    if normalize_images:
        source_img = (source_img - source_img.min()) / (source_img.max() - source_img.min())
        target_img = (target_img - target_img.min()) / (target_img.max() - target_img.min())
    shift, error, diffphase = phase_cross_correlation(source_img, target_img, upsample_factor=100)
    corrected_image = apply_shift(target_img, shift)
    return corrected_image, shift