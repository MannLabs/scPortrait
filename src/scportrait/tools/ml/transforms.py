import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class RandomRotation:
    """Randomly rotate input image in x-degree steps (default is 90).

    Args:
        choices: number of possible rotations
        include_zero: include 0 degree rotation in the choices
    """

    def __init__(self, choices: int = 4, include_zero: bool = True):
        angles = np.linspace(0, 360, choices + 1)
        angles = angles[:-1]

        if not include_zero:
            delta = (360 - angles[-1]) / 2
            angles = angles + delta

        self.choices = angles.tolist()

    def __call__(self, tensor):
        angle = random.choice(self.choices)
        return TF.rotate(tensor, angle)


class GaussianNoise:
    """Add Gaussian noise to the input tensor.

    Args:
        sigma: Standard deviation of the Gaussian noise.
        channels_to_exclude: List of channel indices to exclude from noise addition.
    """

    def __init__(self, sigma: float = 0.1, channels_to_exclude: list[int] | None = None):
        assert sigma > 0, "sigma must be greater than 0."
        self.sigma = sigma
        self.channels = channels_to_exclude or []

    def __call__(self, tensor):
        if self.sigma != 0:
            scale = self.sigma * tensor
            sampled_noise = torch.tensor(0, dtype=torch.float32).repeat(*tensor.size()).normal_() * scale

            # remove noise for masked channels
            if len(tensor.shape) == 3:
                for channel in range(tensor.shape[0]):
                    if channel in self.channels:
                        sampled_noise[channel, :, :] = 0

            tensor = tensor + sampled_noise
        return tensor


class GaussianBlur:
    """Apply a gaussian blur to the input image.

    Args:
        kernel_size: list of kernel sizes to randomly select from
        sigma: tuple of min and max sigma values to randomly select from
    """

    def __init__(self, kernel_size: list[int] | None = None, sigma: tuple[float, float] | None = None):
        if kernel_size is None:
            kernel_size = [1, 1, 1, 1, 5, 5, 7, 9]
        if sigma is None:
            sigma = (0.1, 2.0)

        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, tensor):
        # randomly select a kernel size and sigma to add more variation
        # kernel size of 1 does not affect the image = 50% of the time no blur is added
        # pytorch randomly selects a value from a uniform distribution between (sigma_min, sigma_max)
        kernel_size = random.choice(self.kernel_size)
        sigma = self.sigma
        blur = T.GaussianBlur(kernel_size, sigma)
        return blur(tensor)


class ChannelSelector:
    """select the channel used for prediction.

    Args:
        channels: list of channel indices to keep
        num_channels: number of channels in the input tensor
    """

    def __init__(self, channels: list[int] | None = None, num_channels: int = 5):
        if channels is None:
            channels = list(range(num_channels))
        assert np.max(channels) < num_channels, "highest channel index exceeds channel numb"
        self.channels = channels

    def __call__(self, tensor):
        return tensor[self.channels, :, :]
