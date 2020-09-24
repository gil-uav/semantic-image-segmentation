import logging
from glob import glob
from os import listdir
from os.path import splitext

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import transforms

from utils.pre_processing import (
    ToTensor,
    Rescale,
    RandomFlip,
    RandomColorJitter,
    RandomNoise,
    MaskToClasses,
)


class OrthogonalPhotoDataset(Dataset):
    def __init__(
        self,
        imgs_dir: str,
        masks_dir: str,
        image_suffix: str = "_x",
        mask_suffix: str = "_y",
        augmentation: bool = True,
        resize: int = 256,
    ):
        """
        Parameters
        ----------
        imgs_dir : str
            Directory for images.
        masks_dir : str
            Directory for masks.
        image_suffix :str
            Suffix for images.
        mask_suffix :str
            Suffix for masks.
        augmentation :bool
            Augmentation bool for tran/test sets.
        resize :int
            Rescale size for images.
        """
        self.rescale = resize
        self.augmentation = augmentation
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.mask_suffix = mask_suffix
        self.image_suffix = image_suffix
        self.ids = [
            splitext(file)[0] for file in listdir(imgs_dir) if not file.startswith(".")
        ]
        logging.info(f"Creating dataset with {len(self.ids)} examples")
        self.mapping = {0: 0, 255: 1}
        # self.mapping = {
        # 0: 0,
        # 127: 1,
        # 255: 2
        # }

    def set_augmentation(self, aug: bool = False):
        """
        Sets augmentation to true or false.

        Parameters
        ----------
        aug : bool
            Data Augmentation on or off.
        """
        self.augmentation = aug

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        """
        # Get image and mask
        """
        idx = self.ids[i]
        img_file = glob(self.imgs_dir + idx + ".*")
        msk_file = glob(
            str(self.masks_dir + idx + ".*").replace(
                self.image_suffix, self.mask_suffix
            )
        )

        if len(msk_file) != 1:
            raise AssertionError(
                f"Either no mask or multiple masks found for the ID {idx}: {msk_file}"
            )
        if len(img_file) != 1:
            raise AssertionError(
                f"Either no image or multiple images found for the ID {idx}: {img_file}"
            )
        img_as_img = Image.open(img_file[0]).convert("RGB")
        msk_as_img = Image.open(msk_file[0]).convert("L")

        if img_as_img.size != msk_as_img.size:
            raise AssertionError(
                f"Image and mask {idx} should be the same size, but are {img_as_img.size} and {msk_as_img.size}"
            )

        # Data augmentation
        if self.augmentation:
            transform = transforms.Compose(
                [
                    Rescale(self.rescale),
                    RandomFlip(),
                    RandomColorJitter(),
                    RandomNoise(),  # Normalised
                    MaskToClasses(self.mapping),
                    ToTensor(),
                ]
            )
        else:
            transform = transforms.Compose(
                [Rescale(self.rescale), MaskToClasses(self.mapping), ToTensor()]
            )

        sample = {"image": img_as_img, "mask": msk_as_img}

        tranformed_sample = transform(sample)

        return tranformed_sample
