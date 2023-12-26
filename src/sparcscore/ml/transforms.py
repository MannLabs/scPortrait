import torch
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class RandomRotation(object):
    """
    Randomly rotates input image in 90 degree steps.

    Args:
        choices (int): Number of possible rotations.

    Returns:
        Rotated image.
    """
    def __init__(self, choices=4, include_zero=True):
        angles = np.linspace(0, 360, choices + 1)
        angles = angles[:-1]

        if not include_zero:
            delta = (360 - angles[-1]) / 2
            angles = angles + delta

        self.choices = angles

    def __call__(self, tensor):
        angle = random.choice(self.choices)
        return TF.rotate(tensor, angle)


class GaussianNoise(object):
    """
    Add Gaussian noise to the input image.

    Args:
        sigma (float): Standard deviation of the Gaussian distribution.
        channels (list): List of channel indices to add noise to.

    Returns:
        Image with added Gaussian noise.
    """
    def __init__(self, sigma=0.1, channels_to_exclude=[]):
        self.sigma = sigma
        self.channels = channels_to_exclude

    def __call__(self, tensor):
        if self.sigma != 0:
            scale = self.sigma * tensor
            sampled_noise = (
                    torch.tensor(0, dtype=torch.float32).repeat(*tensor.size()).normal_()
                    * scale
            )

            # remove noise for masked channels
            if len(tensor.shape) == 3:
                for channel in range(tensor.shape[0]):
                    if channel in self.channels:
                        sampled_noise[channel, :, :] = 0

            tensor = tensor + sampled_noise
        return tensor


class GaussianBlur(object):
    """
    Applies Gaussian blur to the input image.

    Args:
        kernel_size (list): List of kernel sizes to randomly select from. 
        sigma (tuple): Tuple of sigma values to randomly select from.
        channels (list): List of channel indices to blur.

    Returns:
        Image with added Gaussian blur.
    """
    def __init__(
            self, kernel_size=[1, 1, 1, 1, 5, 5, 7, 9], sigma=(0.1, 2), channels=[]
    ):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.channels = channels

    def __call__(self, tensor):
        # randomly select a kernel size and sigma to add more variation
        # kernel size of 1 does not affect the image = 50% of the time no blur is added
        # pytorch randomly selects a value from a uniform distribution between (sigma_min, sigma_max)
        kernel_size = random.choice(self.kernel_size)
        sigma = self.sigma
        blur = T.GaussianBlur(kernel_size, sigma)

        # return the corrected image
        return blur(tensor)


class ChannelReducer(object):
    """
    Reduces the number of channels in the input image to 5, 3 or 1 channel.

    5: nuclei_mask, cell_mask, channel_nucleus, channel_cellmask, channel_of_interest
    3: nuclei_mask, cell_mask, channel_of_interest
    1: channel_of_interest

    Args:
        channels (int): Number of channels to keep.
        
    Returns:
        Image with reduced number of channels.
    """
    def __init__(self, channels=5):
        self.channels = channels

    def __call__(self, tensor):
        if self.channels == 1:
            return tensor[[3], :, :]
        elif self.channels == 3:
            return tensor[[0, 1, 3], :, :]
        else:
            return tensor


class ChannelSelector(object):
    """
    Selects the channels of interest from the input image.

    Args:
        channels (list): List of channel indices to select.
        num_channels (int): Number of channels in the input image.
    
    Returns:
        Image with only the selected channels.
    """
    def __init__(self, channels=[0, 1, 2, 3, 4], num_channels=5):
        if not np.max(channels) < num_channels:
            raise ValueError("highest channel index exceeds channel numb")
        self.channels = channels

    def __call__(self, tensor):
        return tensor[self.channels, :, :]


class ImageDownsampler(object):
    """
    Downsamples the image to a given size.

    Args:
        size (int): Desired image size.

    Returns:
        Downsampled image.
    """
    def __init__(self, size):
        self.size = size

    def __call__(self, tensor):
        return TF.resize(tensor, self.size, antialias=True)
