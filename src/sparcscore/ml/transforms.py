import torch
import numpy as np
import random
import torchvision.transforms.functional as TF
import torchvision.transforms as T


class RandomRotation(object):
    """
    Randomly rotate input image in 90 degree steps.
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
    Add gaussian noise to the input image.
    """

    def __init__(self, sigma=0.1, channels=[]):
        self.sigma = sigma
        self.channels = channels

    def __call__(self, tensor):
        if self.sigma != 0:
            scale = self.sigma * tensor
            sampled_noise = (
                torch.tensor(0, dtype=torch.float32).repeat(*tensor.size()).normal_()
                * scale
            )

            # remove noise for masked channels

            for channel in range(tensor.shape[0]):
                if channel not in self.channels:
                    sampled_noise[channel, :, :] = 0

            tensor = tensor + sampled_noise
        return tensor


class GaussianBlur(object):
    """
    Apply a gaussian blur to the input image.
    """

    def __init__(
        self, kernel_size=[1, 1, 1, 1, 5, 5, 7, 9], sigma=(0.1, 0.2), channels=[]
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
    can reduce an imaging dataset dataset to 5, 3 or 1 channel
    5: nuclei_mask, cell_mask, channel_nucleus, channel_cellmask, channel_of_interest
    3: nuclei_mask, cell_mask, channel_of_interestå
    1: channel_of_interestå
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
    select the channel used for prediction.
    """

    def __init__(self, channels=[0, 1, 2, 3, 4], num_channels=5):
        if not np.max(channels) < num_channels:
            raise ValueError("highest channel index exceeds channel numb")
        self.channels = channels

    def __call__(self, tensor):
        return tensor[self.channels, :, :]


class Downsize(object):
    """
    Downsize the image to a given size.
    """

    def __init__(self, size=160):
        self.size = (size, size)

    def __call__(self, tensor):
        return TF.resize(tensor, size=self.size)

