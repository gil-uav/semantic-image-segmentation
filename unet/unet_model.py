"""
Pytorch Ignite U-net module.
"""
import copy
import os
import random
from argparse import ArgumentParser

import kornia
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
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
        self.learning_rate = learning_rate
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

        # Model definition start
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
        # Model definition end

        # Dataset
        self.train_set = None
        self.val_set = None
        self.test_set = None

        # Metrics
        self.train_pres = pl.metrics.Precision(num_classes=self.hparams.n_classes)
        self.train_rec = pl.metrics.Recall(num_classes=self.hparams.n_classes)
        self.train_f1 = pl.metrics.Fbeta(num_classes=self.hparams.n_classes)

        self.val_pres = pl.metrics.Precision(num_classes=self.hparams.n_classes)
        self.val_rec = pl.metrics.Recall(num_classes=self.hparams.n_classes)
        self.val_f1 = pl.metrics.Fbeta(num_classes=self.hparams.n_classes)

        self.test_pres = pl.metrics.Precision(num_classes=self.hparams.n_classes)
        self.test_rec = pl.metrics.Recall(num_classes=self.hparams.n_classes)
        self.test_f1 = pl.metrics.Fbeta(num_classes=self.hparams.n_classes)

        # Logging iterations
        self.train_log = 0
        self.val_log = 0
        self.test_log = 0

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
        """
        Initiates datasets.
        """
        dataset = OrthogonalPhotoDataset(**self.hparams)
        test_dataset = copy.deepcopy(OrthogonalPhotoDataset(**self.hparams))
        test_amount = int(len(dataset) * self.hparams.test_percent / 100)
        val_amount = int(len(dataset) * self.hparams.val_percent / 100)
        train_amount = len(dataset) - val_amount - test_amount

        # Logging iterations
        self.train_log = int(train_amount / 10)
        self.val_log = int(val_amount / 10)
        self.test_log = int(test_amount / 10)

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
        """
        Creates dataloader for the training set.
        """
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=int(os.getenv("WORKERS")),
            pin_memory=True,
        )

    def val_dataloader(self):
        """
        Creates dataloader for the validation set.
        """
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=int(os.getenv("WORKERS")),
            pin_memory=True,
            drop_last=True,
        )

    def test_dataloader(self):
        """
        Creates dataloader for the test set.
        """
        return DataLoader(
            self.test_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=int(os.getenv("WORKERS")),
            pin_memory=True,
            drop_last=True,
        )

    def configure_optimizers(self):
        """
        Configures optimizers and learning rate scheduler.

        Returns
        -------
        dict {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "val_loss",
            }

        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.learning_rate or self.learning_rate)
        )
        lr_scheduler = {
            "scheduler": ReduceLROnPlateau(
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
            ),
            "name": "learning_rate",
            "monitor": "val_loss",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }

    def loss_function(self, logits, target):
        """
        Loss function.
        """
        return self.criterion(logits, target)

    def log_parameters(self):
        """
        Logs parameters as distribution and histograms.
        """
        for tag, value in self.named_parameters():
            tag = tag.replace(".", "/")
            self.logger.experiment.add_histogram(tag, value, self.current_epoch)

    def log_images(self, images, masks, masks_pred, no_imgs: int, step: str):
        """
        Logs example images.
        """
        no_imgs = max(min(no_imgs, self.hparams.batch_size), 0)
        rand_idx = random.sample(range(0, self.hparams.batch_size), no_imgs)
        msk_list = []
        img_list = []

        alpha = 0.5
        for idx in rand_idx:
            onehot = torch.gt(torch.sigmoid(masks_pred[idx]), 0.5)
            msk_grid = torchvision.utils.make_grid([masks[idx], onehot], nrow=1)
            msk_list.append(msk_grid)
            img_grid = torchvision.utils.make_grid(
                [
                    images[idx],
                    torch.clamp(
                        kornia.enhance.add_weighted(
                            src1=images[idx],
                            alpha=1.0,
                            src2=onehot,
                            beta=alpha,
                            gamma=0.0,
                        ),
                        max=1.0,
                    ),
                ],
                nrow=1,
            )
            img_list.append(img_grid)

        mask_grid = torchvision.utils.make_grid(msk_list, nrow=no_imgs)
        self.logger.experiment.add_image(
            "{} - Target vs Predicted".format(step), mask_grid, self.global_step
        )
        image_grid = torchvision.utils.make_grid(img_list, nrow=no_imgs)
        self.logger.experiment.add_image(
            "{} - Image vs Predicted".format(step), image_grid, self.global_step
        )

    def training_step(self, batch, batch_idx):
        """
        Training step.
        """
        images, masks, masks_pred, loss = self.shared_step(batch)

        pred = torch.gt(torch.sigmoid(masks_pred), 0.5).half()
        self.train_pres(pred, masks)
        self.train_rec(pred, masks)
        self.train_f1(pred, masks)

        self.log("train_loss", loss)
        self.log("train_precision", self.train_pres)
        self.log("train_recall", self.train_rec)
        self.log("train_f1", self.train_f1)

        if batch_idx % self.train_log == 0:
            self.log_parameters()
            self.log_images(images, masks, masks_pred, 5, "TRAIN")

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        """
        images, masks, masks_pred, loss = self.shared_step(batch)

        pred = torch.gt(torch.sigmoid(masks_pred), 0.5).half()
        self.val_pres(pred, masks)
        self.val_rec(pred, masks)
        self.val_f1(pred, masks)

        self.log("val_loss", loss)
        self.log("val_precision", self.val_pres)
        self.log("val_recall", self.val_rec)
        self.log("val_f1", self.val_f1)

        if batch_idx % self.val_log == 0:
            self.log_images(images, masks, masks_pred, 5, "VAL")

    def test_step(self, batch, batch_idx):
        """
        Test step.
        """
        images, masks, masks_pred, loss = self.shared_step(batch)

        pred = torch.gt(torch.sigmoid(masks_pred), 0.5).half()
        self.test_pres(pred, masks)
        self.test_rec(pred, masks)
        self.test_f1(pred, masks)

        self.log("test_loss", loss)
        self.log("test_precision", self.test_pres)
        self.log("test_recall", self.test_rec)
        self.log("test_f1", self.test_f1)

        if batch_idx % self.test_log == 0:
            self.log_images(images, masks, masks_pred, 5, "TEST")

    def shared_step(self, batch):
        """
        Runs forward prop and calculates metrics.
        """
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
        loss = self.loss_function(masks_pred, masks)

        return images, masks, masks_pred, loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Argument parser.
        """
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
