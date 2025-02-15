#######################################################
# Unit tests for ../tools/ml/transforms.py
#######################################################

import pytest
import torch

from scportrait.tools.ml.transforms import ChannelSelector, GaussianBlur, GaussianNoise, RandomRotation


@pytest.fixture
def sample_tensor():
    """Fixture to create a sample 3-channel tensor."""
    return torch.rand((3, 64, 64))  # (C, H, W) format


@pytest.fixture
def batch_sample_tensor():
    """Fixture to create a sample batch of 3-channel tensors."""
    return torch.rand((8, 3, 64, 64))  # (N, C, H, W) format


### RandomRotation Tests ###
def test_random_rotation(sample_tensor):
    transform = RandomRotation(choices=4, include_zero=True)
    output = transform(sample_tensor)

    assert output.shape == sample_tensor.shape, "Rotation should not change tensor shape"
    assert isinstance(output, torch.Tensor), "Output must be a torch.Tensor"


def test_random_rotation_excludes_zero(sample_tensor):
    transform = RandomRotation(choices=4, include_zero=False)
    output = transform(sample_tensor)

    assert output.shape == sample_tensor.shape
    assert isinstance(output, torch.Tensor)


### GaussianNoise Tests ###
def test_gaussian_noise(sample_tensor):
    transform = GaussianNoise(sigma=0.1)
    output = transform(sample_tensor)

    assert output.shape == sample_tensor.shape, "Noise should not change shape"
    assert isinstance(output, torch.Tensor), "Output must be a torch.Tensor"
    assert not torch.equal(output, sample_tensor), "Output should be different due to added noise"


def test_gaussian_noise_exclude_channels(sample_tensor):
    transform = GaussianNoise(sigma=0.1, channels_to_exclude=[0, 2])
    output = transform(sample_tensor)

    assert torch.equal(output[0], sample_tensor[0]), "Channel 0 should not have noise"
    assert torch.equal(output[2], sample_tensor[2]), "Channel 2 should not have noise"


def test_gaussian_noise_invalid_sigma():
    with pytest.raises(AssertionError):
        GaussianNoise(sigma=-0.1)  # Negative sigma should raise an error


### GaussianBlur Tests ###
def test_gaussian_blur(sample_tensor):
    transform = GaussianBlur(kernel_size=[3, 5], sigma=(0.5, 1.5))
    output = transform(sample_tensor)

    assert output.shape == sample_tensor.shape, "Blur should not change shape"
    assert isinstance(output, torch.Tensor), "Output must be a torch.Tensor"


def test_gaussian_blur_default_params(sample_tensor):
    transform = GaussianBlur()
    output = transform(sample_tensor)

    assert output.shape == sample_tensor.shape


### ChannelSelector Tests ###
def test_channel_selector(sample_tensor):
    transform = ChannelSelector(channels=[0, 2], num_channels=3)
    output = transform(sample_tensor)

    assert output.shape == (2, 64, 64), "Output shape should match selected channels"


def test_channel_selector_all_channels(sample_tensor):
    transform = ChannelSelector(num_channels=3)
    output = transform(sample_tensor)

    assert torch.equal(output, sample_tensor), "Selecting all channels should return unchanged tensor"


def test_channel_selector_invalid_index():
    with pytest.raises(AssertionError):
        ChannelSelector(channels=[0, 4], num_channels=3)  # Index 4 is out of bounds
