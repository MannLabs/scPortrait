import random

import matplotlib.pyplot as plt
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
        sigma: Strength of the added noise. If a range is provided a random value will be sampled form this range.
        channels_to_exclude: List of channel indices to exclude from noise addition.
        mean: Mean of the Gaussian distribution from which the noise is sampled.
        std: Standard deviation of the Gaussian distribution from which the noise is sampled.
        deep_debug: If True, the input tensor, the sampled noise, and the transformed tensor are plotted for debugging.
    """

    def __init__(
        self,
        sigma: float | tuple[float, float] = 0.1,
        mean: float = 0.5,
        std: float = 1,
        channels_to_exclude: list[int] | None = None,
        deep_debug: bool = False,
    ):
        if channels_to_exclude is None:
            channels_to_exclude = []

        if isinstance(sigma, tuple):
            assert isinstance(sigma[0], float) and isinstance(sigma[1], float), "sigma must be a tuple of two floats."
            assert sigma[0] < sigma[1], "sigma min must be less than sigma max."
            self.sigma_sample = True
            self.sigma_range = sigma
        elif isinstance(sigma, float):
            assert sigma >= 0, "sigma must be a positive float."
            self.sigma_sample = False
            self.sigma = sigma

        self.mean = mean
        self.std = std
        self.channels = channels_to_exclude or []
        self.deep_debug = deep_debug  # can be set to true for debugging purposes

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        # randomly sample sigma from a uniform distribution
        if self.sigma_sample:
            lower, upper = self.sigma_range
            sigma = random.uniform(lower, upper)
        else:
            sigma = self.sigma

        scale = sigma * tensor
        sampled_noise = (
            torch.tensor(0, dtype=torch.float32).repeat(*tensor.size()).normal_(mean=self.mean, std=self.std) * scale
        )

        # Vectorized masking of excluded channels
        if tensor.ndimension() == 3:  # (C, H, W)
            sampled_noise[self.channels, :, :] = 0
        elif tensor.ndimension() == 4:  # (N, C, H, W)
            sampled_noise[:, self.channels, :, :] = 0

        sampled_noise = torch.clamp(sampled_noise, 0, 1)  # clip to 0,1 (don't want to reset the scale of our values)

        if self.deep_debug:
            n_channels = tensor.shape[1]

            fig, axs = plt.subplots(1, n_channels, figsize=(n_channels * 2, 2))
            if n_channels == 1:
                axs.imshow(tensor[0, 0], vmin=0, vmax=1)
                axs.axis("off")
                axs.set_title("Input tensor sampled noise")
            else:
                for i in range(n_channels):
                    axs[i].imshow(tensor[0, i], vmin=0, vmax=1)
                    axs[i].axis("off")
                    axs[i].set_title(f"Input tensor index {i} sampled noise")

            fig, axs = plt.subplots(1, n_channels, figsize=(n_channels * 2, 2))
            if n_channels == 1:
                axs.imshow(sampled_noise[0, 0], vmin=0, vmax=1)
                axs.axis("off")
                axs.set_title("sampled noise")
            else:
                for i in range(n_channels):
                    axs[i].imshow(sampled_noise[0, i], vmin=0, vmax=1)
                    axs[i].axis("off")
                    axs[i].set_title(f"index {i} sampled noise")

        tensor = tensor.add_(sampled_noise)
        tensor = torch.clamp(tensor, 0, 1)  # clip to 0,1 (also after addition of noise)

        if self.deep_debug:
            n_channels = tensor.shape[1]

            fig, axs = plt.subplots(1, n_channels, figsize=(n_channels * 2, 2))
            if n_channels == 1:
                axs.imshow(tensor[0, 0], vmin=0, vmax=1)
                axs.axis("off")
                axs.set_title("Transformed tensor sampled noise")
            else:
                for i in range(n_channels):
                    axs[i].imshow(tensor[0, i], vmin=0, vmax=1)
                    axs[i].axis("off")
                    axs[i].set_title(f"Transformed tensor index {i} sampled noise")

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


class ChannelMultiplier:
    """
    duplicate the input tensor

    Args:
        n_reps: how often to duplicate the tensor
    """

    def __init__(self, n_reps: int = 3):
        self.n_reps = n_reps

    def __call__(self, tensor):
        return torch.zeros((self.n_reps, 1, 1)) + tensor
