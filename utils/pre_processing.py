import random
from random import randint

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from skimage.util import random_noise
from torchvision import transforms


class Rescale():
    """
    Rescale the image in a sample to a given size, returns image as min-max normalized (0,1).

    Parameters
    ----------
    output_size : (tuple or int)
        Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        if not isinstance(output_size, (int, tuple)):
            raise AssertionError
        self.output_size = output_size

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        h, w = image.size
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = image.resize((new_h, new_w))
        msk = mask.resize((new_h, new_w), resample=Image.NEAREST)  # Nearest neighbour

        return {"image": img, "mask": msk}


class ToTensor():
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = transforms.ToTensor()(image)
        # image = torch.from_numpy(np.array(image).transpose((2, 0, 1)))

        return {"image": image, "mask": mask}


class RandomRotate():
    """
    Rotate randomly the image and mask in a sample. (90, 180 or 270 degrees)
    """

    def __init__(self):
        self.rotate = randint(0, 3)

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        if self.rotate == 0:
            # vertical
            image.transpose(Image.ROTATE_90)
            mask.transpose(Image.ROTATE_90)
        elif self.rotate == 1:
            # horizontal
            image.transpose(Image.ROTATE_180)
            mask.transpose(Image.ROTATE_180)
        elif self.rotate == 2:
            image.transpose(Image.ROTATE_270)
            mask.transpose(Image.ROTATE_270)
        else:
            # no effect
            image = image
            mask = mask

        return {"image": image, "mask": mask}


class RandomFlip():
    """
    Flip randomly the image and mask in a sample.
    """

    def __init__(self):
        self.flip = randint(0, 3)

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        if self.flip == 0:
            # vertical
            image.transpose(Image.FLIP_LEFT_RIGHT)
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        elif self.flip == 1:
            # horizontal
            image.transpose(Image.FLIP_TOP_BOTTOM)
            mask.transpose(Image.FLIP_TOP_BOTTOM)
        elif self.flip == 2:
            # horizontally and vertically flip
            image.transpose(Image.FLIP_TOP_BOTTOM)
            mask.transpose(Image.FLIP_TOP_BOTTOM)
            image.transpose(Image.FLIP_LEFT_RIGHT)
            mask.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            # no effect
            image = image
            mask = mask

        return {"image": image, "mask": mask}


class RandomNoise():
    """
    Adds noise randomly to the image in a sample, also applies min-max normalizaiton(0,1)
    due to skimage functions.

    Parameters
    ----------
    noise : int
        [-1] -> Randomly chosen
        [0] Gaussian
        [1] Salt and pepper
        [2] Poisson
        [3] Speckle
        [4] None
    """

    def __init__(self, noise: int = -1):
        if noise == -1:
            self.noise = randint(0, 4)
        else:
            if noise > 0:
                raise AssertionError("Noise override out of bounds.")
            self.noise = noise

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]

        if self.noise == 0:
            # Gaussian noise
            radius = random.uniform(0.01, 1.0)
            image = image.filter(ImageFilter.GaussianBlur(radius=radius))
        elif self.noise == 1:
            # Salt and pepper
            if type(image) != np.ndarray:
                image = np.array(image)
            amount = random.uniform(0.001, 0.05)
            image = random_noise(image, mode="s&p", amount=amount, clip=True)
        elif self.noise == 2:
            # Poisson
            if type(image) != np.ndarray:
                image = np.array(image)
            image = random_noise(image, mode="poisson", clip=True)
        elif self.noise == 3:
            # Speckle
            if type(image) != np.ndarray:
                image = np.array(image)
            mean = 0
            var = 0.1
            image = random_noise(image, mode="speckle", mean=mean, var=var, clip=True)

        return {"image": image, "mask": mask}


class RandomBrightness():
    """
    Changes brightness randomly to the image in a sample.

    Parameters
    ----------
    low : int
        Lower value for brightness adjustment.
    high : int
        High value for brightness adjustment.
    """

    def __init__(self, low: int = 0.5, high: int = 1.5):
        self.factor = random.uniform(low, high)

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(self.factor)
        return {"image": image, "mask": mask}


class RandomSharpness():
    """
    Changes the sharpness randomly to the image in a sample.

    Parameters
    ----------
    low : int
        Lower value for brightness adjustment.
    high : int
        High value for brightness adjustment.
    """

    def __init__(self, low: int = 0.5, high: int = 1.5):
        self.factor = random.uniform(low, high)

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(self.factor)
        return {"image": image, "mask": mask}


class RandomContrast():
    """
    Changes Contrast randomly to the image in a sample.

    Parameters
    ----------
    low : int
        Lower value for brightness adjustment.
    high : int
        High value for brightness adjustment.
    """

    def __init__(self, low: int = 0.5, high: int = 1.5):
        self.factor = random.uniform(low, high)

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(self.factor)
        return {"image": image, "mask": mask}


class MaskToClasses():
    """
    Converts mask images to tensors with class indices from 0 to (number of colors) - 1.

    Parameters
    ----------
    mapping: dict or None
        Mapping of colors to classes.
        mapping = {0: 0, 127: 1, 255: 2}
    """

    def __init__(self, mapping):
        if not isinstance(mapping, dict):
            raise AssertionError
        self.mapping = mapping

    def __call__(self, sample):
        image, mask = sample["image"], sample["mask"]
        # Multi-class
        if len(self.mapping) > 2:
            mask = torch.from_numpy(np.array(mask))
            for k in self.mapping:
                mask[mask == k] = self.mapping[k]
            return {"image": image, "mask": mask}
        mask = transforms.ToTensor()(mask)
        return {"image": image, "mask": mask}


if __name__ == "__main__":
    pass
