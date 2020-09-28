import os
from argparse import ArgumentParser
from random import randint

import kornia
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision
from pytorch_lightning.metrics.functional import f1_score, recall, precision, accuracy
from torch.utils.data import random_split, DataLoader

from unet.unet_modules import DoubleConvolution, Down, Up, OutConvolution
from utils.dataset import OrthogonalPhotoDataset


class UNet(pl.LightningModule):
    """
    Basic U-net structure as described in : O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional networks
    for biomedical image segmentation.” 2015.
    """

    def __init__(
        self,
        n_channels: int = 3,
        n_classes: int = 1,
        bilinear=True,
        lr: float = 0.0001,
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
        self.lr = lr
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
        val_amount = int(len(dataset) * self.hparams.val_percent / 100)
        train_amount = len(dataset) - val_amount

        # Development-mode
        if not os.getenv("PROD"):
            # Fix the generator for reproducible results
            self.train_set, self.val_set = random_split(
                dataset,
                [train_amount, val_amount],
                generator=torch.Generator().manual_seed(42),
            )
        # Production-mode
        else:
            # Random generator
            self.train_set, self.val_set = random_split(
                dataset, [train_amount, val_amount]
            )

        # Disable augmentation for validation set.
        self.val_set.dataset.set_augmentation(False)

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=int(os.getenv("WORKERS", 0)),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            num_workers=int(os.getenv("WORKERS", 0)),
            pin_memory=True,
            drop_last=True,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=(self.lr or self.learning_rate)
        )
        return [optimizer]

    def loss_funciton(self, input, target):
        return self.criterion(input, target)

    def training_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]
        if images.shape[1] != self.hparams.n_channels:
            raise AssertionError(
                f"Network has been defined with {self.hparams.n_channels} input channels, "
                f"but loaded images have {images.shape[1]} channels. Please check that "
                "the images are loaded correctly."
            )

        masks = (
            masks.type(torch.float32)
            if self.hparams.n_classes == 1
            else masks.type(torch.long)
        )

        masks_pred = self(images)  # Forward pass
        loss = self.loss_funciton(masks_pred, masks)
        result = pl.TrainResult(minimize=loss)
        result.log("train_loss", loss, on_epoch=True, sync_dist=True)

        return result

    def validation_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]
        if images.shape[1] != self.hparams.n_channels:
            raise AssertionError(
                f"Network has been defined with {self.n_channels} input channels, "
                f"but loaded images have {images.shape[1]} channels. Please check that "
                "the images are loaded correctly."
            )

        masks = (
            masks.type(torch.float32)
            if self.hparams.n_classes == 1
            else masks.type(torch.long)
        )

        masks_pred = self(images)  # Forward pass
        loss = self.loss_funciton(masks_pred, masks)
        result = pl.EvalResult(loss, checkpoint_on=loss)
        result.log("val_loss", loss, on_epoch=True, sync_dist=True)
        if batch_idx == 0:
            rand_idx = randint(0, self.hparams.batch_size - 1)
            onehot = torch.sigmoid(masks_pred[rand_idx]) > 0.5
            for tag, value in self.named_parameters():
                tag = tag.replace(".", "/")
                self.logger.experiment.add_histogram(tag, value, self.current_epoch)
            mask_grid = torchvision.utils.make_grid([masks[rand_idx], onehot], nrow=2)
            self.logger.experiment.add_image(
                "Target vs Predicted", mask_grid, self.current_epoch
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
                "Image vs Predicted", image_grid, self.current_epoch
            )
        pred = (torch.sigmoid(masks_pred) > 0.5).float()
        f1 = f1_score(pred, masks, self.hparams.n_classes + 1)
        rec = recall(pred, masks, self.hparams.n_classes + 1)
        pres = precision(pred, masks, self.hparams.n_classes + 1)
        acc = accuracy(pred, masks, self.hparams.n_classes + 1)
        result.log("f1", f1, on_epoch=True)
        result.log("recall", rec, on_epoch=True)
        result.log("precision", pres, on_epoch=True)
        result.log("accuracy", acc, on_epoch=True)

        return result

    def test_step(self, batch, batch_idx):
        images, masks = batch["image"], batch["mask"]
        if images.shape[1] != self.n_channels:
            raise AssertionError(
                f"Network has been defined with {self.n_channels} input channels, "
                f"but loaded images have {images.shape[1]} channels. Please check that "
                "the images are loaded correctly."
            )

        masks = (
            masks.type(torch.float32)
            if self.hparams.n_classes == 1
            else masks.type(torch.long)
        )

        masks_pred = self(images)  # Forward pass
        loss = self.loss_funciton(masks_pred, masks)
        result = pl.EvalResult(loss)
        result.log("test_loss", loss)

        return result

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
