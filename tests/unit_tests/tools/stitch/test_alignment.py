import numpy as np
import pytest
from scipy.ndimage import fourier_shift

from scportrait.tools.stitch._utils.alignment import apply_shift, get_aligned_img


@pytest.fixture
def sample_images():
    """Generates synthetic images for testing."""
    rng = np.random.Generator(np.random.PCG64(42))  # Using PCG64 algorithm
    img_size = (100, 100)
    source_img = rng.random(img_size)  # Use the Generator's random method
    target_img = np.copy(source_img)
    known_shift = np.array([5, -3])  # (y, x) shift
    target_img = np.fft.ifftn(fourier_shift(np.fft.fftn(target_img), -known_shift)).real

    return source_img, target_img, known_shift


def test_apply_shift(sample_images):
    """Test that apply_shift correctly applies a Fourier shift."""
    source_img, target_img, known_shift = sample_images
    shifted_back = apply_shift(target_img, known_shift)
    assert np.allclose(source_img, shifted_back, atol=1e-5)


def test_get_aligned_img(sample_images):
    """Test that get_registered_img correctly finds and applies the shift."""
    source_img, target_img, known_shift = sample_images
    registered_img, computed_shift = get_aligned_img(source_img, target_img)
    assert np.allclose(computed_shift, known_shift, atol=0.5), f"Expected {known_shift}, but got {computed_shift}"
    assert np.allclose(registered_img, source_img, atol=1e-2)
