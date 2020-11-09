"""
Pytorch Ignite U-net module.
"""
import copy
import os
from argparse import ArgumentParser
from random import randint

import kornia
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.metrics.functional import f1_score, precision, recall
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, random_split

from unet.unet_modules import DoubleConvolution, Down, OutConvolution, Up
from utils.dataset import OrthogonalPhotoDataset


class UNet(pl.LightningModule):
    """Basic U-net structure as described in : O. Ronneberger, P. Fischer, and T. Brox,
    “U-net: Convolutional networks for biomedical image segmentation.” 2015.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear=True,
        learning_rate: float = 0.0001,
        **kwargs,
    ):
        """

        Parameters
        ----------
        n_channels : int
            Number of channels in input-data.
        n_classes : int
            Number of classes to segment.
        bilinear : bool
            Bilinear interpolation in upsampling(default)
        """
        super().__init__()
        self.learnig_rate = learning_rate
        self.save_hyperparameters()

        self.example_input_array = torch.randn(
            self.hparams.batch_size,
            n_channels,
            self.hparams.image_size,
            self.hparams.image_size,
        )

        if self.hparams.n_classes > 1:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        factor = 2 if bilinear else 1

        self.in_conv = DoubleConvolution(n_channels, 64)
        self.down_conv_1 = Down(64, 128)
        self.down_conv_2 = Down(128, 256)
        self.down_conv_3 = Down(256, 512)
        self.down_conv_4 = Down(512, 1024 // factor)

        self.up_conv_1 = Up(1024, 512 // factor, bilinear)
        self.up_conv_2 = Up(512, 256 // factor, bilinear)
        self.up_conv_3 = Up(256, 128 // factor, bilinear)
        self.up_conv_4 = Up(128, 64, bilinear)
        self.out_conv = OutConvolution(64, n_classes)

        self.train_set = None
        self.val_set = None
        self.test_set = None

    def forward(self, x):
        """
        Feed-forward function.

        Parameters
        ----------
        x : torch.tensor
            Input data
        """
        x1 = self.in_conv(x)
        x2 = self.down_conv_1(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.down_conv_3(x3)
        x5 = self.down_conv_4(x4)

        x = self.up_conv_1(x5, x4)
        x = self.up_conv_2(x, x3)
        x = self.up_conv_3(x, x2)
        x = self.up_conv_4(x, x1)
        out = self.out_conv(x)

        return out

    def setup(self, stage):
        dataset = OrthogonalPhotoDataset(**self.hparams)
        test_dataset = copy.deepcopy(OrthogonalPhotoDataset(**self.hparams))
        test_amount = int(len(dataset) * self.hparams.test_percent / 100)
        val_amount = int(len(dataset) * self.hparams.val_percent / 100)
        train_amount = len(dataset) - val_amount - test_amount

        # Development-mode
        if not os.getenv("PROD"):
            # Fix the generator for reproducible results
            self.train_set, self.val_set, self.test_set = random_split(
                dataset,
                [train_amount, val_amount, test_amount],
                generator=torch.Generator().manual_seed(42),
            )
        # Production-mode
        else:
            # Random generator
            self.train_set, self.val_set, self.test_set = random_split(
                dataset, [train_amount, val_amount, test_amount]
            )

        # Disable augmentation for validation set.
        self.val_set.dataset = test_dataset
        self.test_set.dataset = test_dataset
        self.val_set.dataset.set_augmentation(False)
        self.test_set.dataset.set_augmentation(False)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=int(os.getenv("WORKERS")),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=int(os.getenv("WORKERS")),
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=int(os.getenv("WORKERS")),
            pin_memory=True,
            drop_last=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.learnig_rate or self.learning_rate)
        )
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.01,
            patience=10,
            threshold=0.0001,
            threshold_mode="rel",
            cooldown=0,
            min_lr=0,
            eps=1e-08,
            verbose=False,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
        }

    def loss_funciton(self, logits, target):
        return self.criterion(logits, target)

    def logg_images(self, images, masks, masks_pred, step: str = ""):
        rand_idx = randint(0, self.hparams.batch_size - 1)
        onehot = torch.gt(torch.sigmoid(masks_pred[rand_idx]), 0.5)

        for tag, value in self.named_parameters():
            tag = tag.replace(".", "/")
            self.logger.experiment.add_histogram(tag, value, self.current_epoch)
        mask_grid = torchvision.utils.make_grid([masks[rand_idx], onehot], nrow=2)
        self.logger.experiment.add_image(
            "{} - Target vs Predicted".format(step), mask_grid, self.current_epoch
        )
        alpha = 0.5
        image_grid = torchvision.utils.make_grid(
            [
                images[rand_idx],
                torch.clamp(
                    kornia.enhance.add_weighted(
                        src1=images[rand_idx],
                        alpha=1.0,
                        src2=onehot,
                        beta=alpha,
                        gamma=0.0,
                    ),
                    max=1.0,
                ),
            ]
        )
        self.logger.experiment.add_image(
            "{} - Image vs Predicted".format(step), image_grid, self.current_epoch
        )

    def training_step(self, batch, batch_idx):
        images, masks, masks_pred, loss = self.shared_step("train", batch)
        if batch_idx == 0:
            self.logg_images(images, masks, masks_pred, "TRAIN")

        return loss

    def validation_step(self, batch, batch_idx):
        images, masks, masks_pred, _ = self.shared_step("val", batch)
        if batch_idx == 0:
            self.logg_images(images, masks, masks_pred, "VAL")

    def test_step(self, batch):
        images, masks, masks_pred, _ = self.shared_step("test", batch)
        self.logg_images(images, masks, masks_pred, "TEST")

    def shared_step(self, step, batch):
        images, masks = batch["image"], batch["mask"]

        if images.shape[1] != self.hparams.n_channels:
            raise AssertionError(
                f"Network has been defined with {self.n_channels} input channels, "
                f"but loaded images have {images.shape[1]} channels. Please check that "
                "the images are loaded correctly."
            )
        masks = (
            masks.type(torch.half)
            if self.hparams.n_classes == 1
            else masks.type(torch.long)
        )
        masks_pred = self(images)
        loss = self.loss_funciton(masks_pred, masks)

        pred = torch.gt(torch.sigmoid(masks_pred), 0.5).half()
        f1 = f1_score(pred, masks)
        rec = recall(pred, masks)
        pres = precision(pred, masks)

        self.log("{}_loss".format(step), loss, sync_dist=True)
        self.log("{}_f1".format(step), f1, on_epoch=True)
        self.log("{}_recall".format(step), rec, on_epoch=True)
        self.log("{}_precision".format(step), pres, on_epoch=True)

        return images, masks, masks_pred, loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "-ch",
            "--n-channels",
            dest="n_channels",
            type=int,
            metavar="NCHS",
            default=3 if not os.getenv("N_CHANNELS") else os.getenv("N_CHANNELS"),
            help="Number of channels in input.",
        )
        parser.add_argument(
            "-cl",
            "--n-classes",
            dest="n_classes",
            type=int,
            metavar="NCLS",
            default=1 if not os.getenv("N_CLASSES") else os.getenv("N_CLASSES"),
            help="Number of classes to classify.",
        )
        parser.add_argument(
            "-lr",
            "--learning-rate",
            dest="lr",
            type=float,
            metavar="LR",
            default=0.0001 if not os.getenv("LRN_RATE") else os.getenv("LRN_RATE"),
            help="Number of classes to classify.",
        )
        parser.add_argument(
            "-b",
            "--bilinear",
            dest="bilinear",
            type=bool,
            default=True if not os.getenv("BILINEAR") else os.getenv("BILINEAR"),
            help="To use bilinear up-scaling i in the model.",
        )
        parser.add_argument(
            "-is",
            "--image-size",
            dest="image_size",
            type=int,
            metavar="IS",
            default=512 if not os.getenv("IMG_SIZE") else os.getenv("IMG_SIZE"),
            help="Image height value for resizing.",
        )
        parser.add_argument(
            "-valp",
            "--validation-percent",
            metavar="VALP",
            dest="val_percent",
            type=float,
            default=10.0 if not os.getenv("VAL_PERC") else os.getenv("VAL_PERC"),
            help="How much of dataset to be used as validation set.",
        )
        parser.add_argument(
            "-tesp",
            "--testing-percent",
            metavar="TESP",
            dest="test_percent",
            type=float,
            default=10.0 if not os.getenv("TEST_PERC") else os.getenv("TEST_PERC"),
            help="How much of dataset to be used as testing set.",
        )
        parser.add_argument(
            "-bs",
            "--batch-size",
            dest="batch_size",
            metavar="BS",
            type=int,
            default=1 if not os.getenv("BATCH_SIZE") else os.getenv("BATCH_SIZE"),
            help="Batch size",
        )
        parser.add_argument(
            "-dp",
            "--data-path",
            metavar="DP",
            dest="data_path",
            type=str,
            default="data" if not os.getenv("DIR_DATA") else os.getenv("DIR_DATA"),
            help="Load model from a .pth file",
        )

        return parser
