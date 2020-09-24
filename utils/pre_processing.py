import numbers
import random
from random import randint

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from skimage.util import random_noise
from torchvision import transforms


def _check_sample(sample: dict):
    if isinstance(sample, dict):
        if len(sample) != 2:
            raise ValueError(
                "Sample must contain image and mask: {'image': image, 'mask': mask}"
            )
    else:
        raise TypeError("Sample must be a dict like: {'image': image, 'mask': mask}")

    return sample


class Rescale(torch.nn.Module):
    """
    Rescale the image in a sample to a given size, returns image as min-max normalized (0,1).

    Parameters
    ----------
    output_size : (tuple or int)
        Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.

    Returns
    ----------
    dict
        Returns sample as {'image': torch.tensor ,'mask': torch.tensor}
    """

    def __init__(self, output_size):
        if not isinstance(output_size, (int, tuple)):
            raise AssertionError
        self.output_size = output_size

    def __call__(self, sample):
        sample = _check_sample(sample)
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


class ToTensor(torch.nn.Module):
    """
    Convert ndarrays in sample to Tensors.

    Returns
    ----------
    dict
        Returns sample as {'image': torch.tensor ,'mask': torch.tensor}
    """

    def __call__(self, sample):
        sample = _check_sample(sample)
        image, mask = sample["image"], sample["mask"]

        image = transforms.ToTensor()(image)
        return {"image": image, "mask": mask}


class RandomRotate(torch.nn.Module):
    """
    Rotate randomly the image and mask in a sample. (90, 180 or 270 degrees)

    Returns
    ----------
    dict
        Returns sample as {'image': PIL.Image.Image,'mask': PIL.Image.Image}
    """

    def __init__(self):
        self.rotate = randint(0, 3)

    def __call__(self, sample):
        sample = _check_sample(sample)
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


class RandomFlip(torch.nn.Module):
    """
    Flip randomly the image and mask in a sample.

    Returns
    ----------
    dict
        Returns sample as {'image': PIL.Image.Image,'mask': PIL.Image.Image}
    """

    def __init__(self):
        self.flip = randint(0, 3)

    def __call__(self, sample):
        sample = _check_sample(sample)
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


class RandomNoise(torch.nn.Module):
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

    Returns
    ----------
    dict
        Returns sample as {'image': PIL.Image.Image,'mask': PIL.Image.Image}
    """

    def __init__(self, noise: int = -1):
        super().__init__()
        self.noise = self._check_input(noise)

    @torch.jit.unused
    def _check_input(self, value):
        if isinstance(value, int):
            if 0 > value > 4:
                raise ValueError(
                    "Noise override out of bounds. Value must be between [0,4] and is {}".format(
                        value
                    )
                )
            if value == -1:
                value = randint(0, 4)

        else:
            raise TypeError("Value must be int from 0 to 4.")

        return value

    def __call__(self, sample):
        sample = _check_sample(sample)
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


class RandomColorJitter(torch.nn.Module):
    """
    Randomly change the brightness, contrast and saturation of an image.

    Parameters
    ----------
    brightness : (float or tuple of float (min, max))
        How much to jitter brightness
    contrast : (float or tuple of float (min, max))
        How much to jitter contrast
    saturation : (float or tuple of float (min, max))
        How much to jitter saturation

    Returns
    ----------
    dict
        Returns sample as {'image': PIL.Image.Image,'mask': PIL.Image.Image}
    """

    def __init__(self, brightness=0, contrast=0, saturation=0):
        self.brightness = self._check_input(brightness, "brightness")
        self.contrast = self._check_input(contrast, "contrast")
        self.saturation = self._check_input(saturation, "saturation")

    @torch.jit.unused
    def _check_input(
        self, value, name, center=1, bound=(0, float("inf")), clip_first_on_zero=True
    ):
        if isinstance(value, numbers.Number):
            if value < 0:
                raise ValueError(
                    "If {} is a single number, it must be non negative.".format(name)
                )
            value = [center - float(value), center + float(value)]
            if clip_first_on_zero:
                value[0] = max(value[0], 0.0)
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not bound[0] <= value[0] <= value[1] <= bound[1]:
                raise ValueError("{} values should be between {}".format(name, bound))
        else:
            raise TypeError(
                "{} should be a single number or a list/tuple with lenght 2.".format(
                    name
                )
            )
        # if value is 0 or (1., 1.) for brightness/contrast/saturation
        if value[0] == value[1] == center:
            value = None
        return value

    def __call__(self, sample):
        sample = _check_sample(sample)
        image, mask = sample["image"], sample["mask"]
        enhancements = []

        if self.brightness is not None:
            brightness_factor = random.uniform(self.brightness[0], self.brightness[1])
            enhancements.append((brightness_factor, ImageEnhance.Brightness(image)))
        if self.contrast is not None:
            contrast_factor = random.uniform(self.contrast[0], self.contrast[1])
            enhancements.append((contrast_factor, ImageEnhance.Brightness(image)))
        if self.saturation is not None:
            saturation_factor = random.uniform(self.saturation[0], self.saturation[1])
            enhancements.append((saturation_factor, ImageEnhance.Brightness(image)))

        random.shuffle(enhancements)
        for transform in enhancements:
            factor = transform[0]
            enhancer = transform[1]
            image = enhancer.enhance(factor)

        return {"image": image, "mask": mask}


class MaskToClasses(torch.nn.Module):
    """
    Converts mask images to tensors with class indices from 0 to (number of colors) - 1.

    Parameters
    ----------
    mapping: dict or None
        Mapping of colors to classes.
        mapping = {0: 0, 127: 1, 255: 2}
        I.e {class: color}

    Returns
    ----------
    dict
        Returns sample as {'image': PIL.Image.Image,'mask': PIL.Image.Image}
    """

    def __init__(self, mapping):
        self.mapping = self._check_input(mapping)

    @torch.jit.unused
    def _check_input(
        self,
        value,
    ):
        if isinstance(value, dict):
            if len(value) < 2:
                raise ValueError(
                    "If its a single class, it must map true and false colors. I.e dict must contain 2 colors."
                )
            if any(0 > c > 255 for c in value.keys()):
                raise ValueError("Colors must be from 0 to 255 and be of type int.")
            if any(0 > c >= len(value) for c in value.values()):
                raise ValueError(
                    "Classes must range from 0 to n_classes and be of type int."
                )
        else:
            raise TypeError(
                "Mapping should be a dictionary with color: class. I.e {collor_1: 0, collor_2: 1}"
            )

        return value

    def __call__(self, sample):
        sample = _check_sample(sample)
        image, mask = sample["image"], sample["mask"]
        # Multi-class
        if len(self.mapping) > 2:
            mask = torch.from_numpy(np.array(mask))
            for k in self.mapping:
                mask[mask == k] = self.mapping[k]
            return {"image": image, "mask": mask}
        mask = transforms.ToTensor()(mask)
        return {"image": image, "mask": mask}
