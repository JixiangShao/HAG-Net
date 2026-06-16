import numpy as np
import random
import cv2
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from PIL import ImageEnhance, ImageFilter


def pad_if_smaller(img, size, fill=0):
    # If the minimum side length of the image is smaller than the given size, pad the image with fill value
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize(object):
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        # Pass integer size, which scales the shorter side of the image to target size
        image = F.resize(image, size)
        # Note the interpolation mode: InterpolationMode.NEAREST is available after torchvision 0.9.0
        # For older versions, use PIL.Image.NEAREST instead
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomVerticalFlip(object):
    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.vflip(image)
            target = F.vflip(target)
        return image, target

class RandomContrast(object):
    """Randomly adjust image contrast"""
    def __init__(self, contrast_factor=(0.8, 1.2)):
        """
        Args:
            contrast_factor: Range of contrast adjustment factor, e.g. (0.8, 1.2) means sample randomly between 0.8 and 1.2
        """
        self.contrast_factor = contrast_factor

    def __call__(self, image, target):
        if random.random() < 0.5:  # Apply contrast adjustment with 50% probability
            factor = random.uniform(*self.contrast_factor)
            image = F.adjust_contrast(image, factor)
        return image, target


class RandomGamma(object):
    """Random gamma correction"""
    def __init__(self, gamma_limit=(0.7, 1.5)):
        """
        Args:
            gamma_limit: Range of gamma value, e.g. (0.7, 1.5) means sample randomly between 0.7 and 1.5
        """
        self.gamma_limit = gamma_limit

    def __call__(self, image, target):
        if random.random() < 0.5:  # Apply gamma correction with 50% probability
            gamma = random.uniform(*self.gamma_limit)
            image = F.adjust_gamma(image, gamma)
        return image, target
class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target