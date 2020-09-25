import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from dotenv import load_dotenv
from pytorch_lightning.metrics.functional import (
    accuracy,
    f1_score,
    precision,
    recall,
    iou,
)
from torch import optim
from torch.backends import cudnn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from unet.unet_model import UNet
from utils.dataset import OrthogonalPhotoDataset

load_dotenv(verbose=True)

dir_img = os.getenv("DIR_IMG", "data/imgs/")
dir_mask = os.getenv("DIR_MASK", "data/masks/")
dir_checkpoint = os.getenv("DIR_CHECKPOINTS", "checkpoints/")


def train_net(
    model: nn.Module,
    dev: torch.device,
    epochs: int = 5,
    batch_size: int = 1,
    lr: float = 0.001,
    val_percent: float = 0.1,
    save_cp: bool = True,
    image_size: int = 256,
    val_int: int = 1,
):
    """
    Trains the network with the custom OrthogonalDataset.
    Remember to set mapping and classes in the dataset.

    Parameters
    ----------
    model : nn.Module
        Pytorch model/network.
    dev : torch.device
        CPU or GPU
    epochs : int
        Number of Epochs
    batch_size : int
        Batch size
    lr : float
        Learning rate
    val_percent : float
        Percentage of the dataset for validation runs
    save_cp : bool
        Save checkpoints
    image_size : int
        Image height resize value
    val_int : int
        Validation interval, i.e 1 -> every Epoch.
    """

    """
    SETUP
    """
    dataset = OrthogonalPhotoDataset(dir_img, dir_mask)
    val_amount = int(len(dataset) * val_percent)
    train_amount = len(dataset) - val_amount
    best_loss = 1e10

    # Development-mode
    if not os.getenv("PROD"):
        # Fix the generator for reproducible results
        train, val = random_split(
            dataset,
            [train_amount, val_amount],
            generator=torch.Generator().manual_seed(42),
        )
    # Production-mode
    else:
        # Random generator
        train, val = random_split(dataset, [train_amount, val_amount])

    # Disable augmentation for validation set.
    val.dataset.set_augmentation(False)

    train_loader = DataLoader(
        train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(os.getenv("WORKERS", 0)),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(os.getenv("WORKERS", 0)),
        pin_memory=True,
        drop_last=True,
    )

    # Tensorboad writer
    writer = SummaryWriter(comment=f"LR_{lr}_BS_{batch_size}")

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {train_amount}
        Validation size: {val_amount}
        Checkpoints:     {save_cp}
        Device:          {dev.type}
        Image height:    {image_size}px
    """
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )
    # Scheduler, reduce lr if no loss decrease over 5 epocs.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min" if model.n_classes > 1 else "max", patience=5
    )

    # Loss function
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    model = nn.DataParallel(
        model, device_ids=list(range(torch.cuda.device_count()))
    ).cuda()

    """
    MAIN TRAINING LOOP
    """
    for epoch in range(epochs):

        train_step(
            model,
            train_loader,
            criterion,
            optimizer,
            train_amount,
            epoch,
            epochs,
            writer,
            dev,
        )

        # Validation
        if epoch % val_int == 0:

            test_metrics = val_step(
                model, val_loader, criterion, optimizer, writer, epoch
            )

            for key, val in zip(test_metrics.keys(), test_metrics.values()):
                logging.info("Validation {}: {}".format(key, val))
                writer.add_scalar("{}/test".format(key), val, epoch)

        scheduler.step(test_metrics["Loss"])

        # Save checkpoint
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            if test_metrics["Loss"] < best_loss:
                best_loss = test_metrics["Loss"]
                torch.save(model.state_dict(), dir_checkpoint + "BEST_CP.pth")
            torch.save(model.state_dict(), dir_checkpoint + f"CP_epoch{epoch + 1}.pth")
            logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()


def train_step(
    model,
    train_loader,
    criterion,
    optimizer,
    train_amount,
    epoch,
    epochs,
    writer,
    dev,
):

    train_loss = 0
    model.train()

    with tqdm(
        total=train_amount, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
    ) as p_bar:

        for idx, batch in enumerate(train_loader):
            images = batch["image"]
            masks = batch["mask"]

            if images.shape[1] != model.module.n_channels:
                raise AssertionError(
                    f"Network has been defined with {model.module.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

            # Same as optimizer.zero_grad() but more efficient.
            for param in model.parameters():
                param.grad = None

            # To device
            mask_type = torch.float32 if model.module.n_classes == 1 else torch.long
            images = images.to(device=dev, dtype=torch.float32, non_blocking=True)
            masks = masks.to(device=dev, dtype=mask_type, non_blocking=True)

            # Automatic mixed precision
            with autocast():
                masks_pred = model(images)  # Forward pass
                loss = criterion(masks_pred, masks)  # Compute loss function

            p_bar.set_postfix(**{"Batch loss": loss.item()})

            train_loss += loss.item()

            # Backward pass, also Synchronizes GPUs
            loss.backward()
            # To avoid the exploding gradients problem
            nn.utils.clip_grad_value_(model.parameters(), 0.1)

            optimizer.step()

            p_bar.update(images.shape[0])

        if epoch == 0:
            writer.add_graph(model.module, images)

        writer.add_scalar("Loss/train", train_loss, epoch)


def val_step(model, val_loader, criterion, optimizer, writer, epoch):
    logging.info("\nRunning test evaluation.")

    for tag, value in model.named_parameters():
        tag = tag.replace(".", "/")
        writer.add_histogram("weights/" + tag, value.data.cpu().numpy(), epoch)
        writer.add_histogram("grads/" + tag, value.grad.data.cpu().numpy(), epoch)

    # Run test evaluation
    model.eval()  # Turn off training layers

    mask_type = torch.float32 if model.module.n_classes == 1 else torch.long
    val_amount = len(val_loader)  # the number of batch
    _accuracy = _f1 = _precision = _recall = _iou = _loss = 0

    with tqdm(
        total=val_amount, desc="Validating", unit="batch", position=0, leave=True
    ) as p_bar:
        for batch in val_loader:
            images, masks = batch["image"], batch["mask"]
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=mask_type)

            # Automatic mixed precision(Hopefully to float16 to utilize tensor cores)
            with autocast(enabled=True):
                with torch.no_grad():  # Disable gradient calculation
                    masks_pred = model(images)
                    _loss += criterion(masks_pred, masks).item()

            pred = torch.sigmoid(masks_pred)
            pred = (pred > 0.5).float()

            # Calculate metrics
            num_classes = model.module.n_classes + 1
            _accuracy += accuracy(pred, masks, num_classes=num_classes).item()
            _f1 += f1_score(pred, masks, num_classes=num_classes).item()
            _precision += precision(pred, masks, num_classes=num_classes).item()
            _recall += recall(pred, masks, num_classes=num_classes).item()
            _iou += iou(pred, masks, num_classes=num_classes).item()

            p_bar.set_postfix(
                **{
                    "L": _loss / val_amount,
                    "A": _accuracy / val_amount,
                    "F1": _f1 / val_amount,
                    "P": _precision / val_amount,
                    "R": _recall / val_amount,
                    "IOU": _iou / val_amount,
                }
            )
            p_bar.update()

    writer.add_images("Images", images, epoch)
    if model.module.n_classes == 1:
        writer.add_images("True masks", masks, epoch)
        writer.add_images(
            "Predicted masks",
            torch.sigmoid(masks_pred) > 0.5,
            epoch,
        )

    model.train()  # Turn on training layers
    writer.add_scalar("Learning rate", optimizer.param_groups[0]["lr"], epoch)

    test_metrics = {
        "Loss": _loss / val_amount,
        "Accuracy": _accuracy / val_amount,
        "F1": _f1 / val_amount,
        "Precision": _precision / val_amount,
        "Recall": _recall / val_amount,
        "IOU": _iou / val_amount,
    }

    return test_metrics


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        nargs="?",
        default=5 if not os.getenv("EPOCHS") else os.getenv("EPOCHS"),
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=1 if not os.getenv("BATCH_SIZE") else os.getenv("BATCH_SIZE"),
        help="Batch size",
        dest="batchsize",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.0001 if not os.getenv("LRN_RATE") else os.getenv("LRN_RATE"),
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-f",
        "--load",
        metavar="LD",
        dest="load",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "-v",
        "--validation",
        dest="val",
        metavar="VL",
        nargs="?",
        type=float,
        default=10.0 if not os.getenv("VAL_PERC") else os.getenv("VAL_PERC"),
        help="Percent of the data that is used as validation (0-100)",
    )
    parser.add_argument(
        "-is",
        "--image-size",
        dest="img_size",
        nargs="?",
        type=int,
        default=256 if not os.getenv("IMG_SIZE") else os.getenv("IMG_SIZE"),
        help="Image height value for resizing",
    )
    parser.add_argument(
        "-vi",
        "--validation-interval",
        metavar="VI",
        dest="val_int",
        nargs="?",
        type=int,
        default=1 if not os.getenv("VAL_INT") else os.getenv("VAL_INT"),
        help="Validation interval, i.e 1 -> every Epocch.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = get_args()

    if bool(os.getenv("PROD")):
        logging.info("Training i production mode, disabling all debugging APIs")
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(enabled=False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
    else:
        logging.info("Training i development mode, debugging APIs active.")
        torch.autograd.set_detect_anomaly(True)
        torch.autograd.profiler.profile(
            enabled=True, use_cuda=True, record_shapes=True, profile_memory=True
        )
        torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")

    model = UNet(
        n_channels=int(os.getenv("N_CHANNELS")),
        n_classes=int(os.getenv("N_CLASSES")),
        bilinear=bool(os.getenv("BILINEAR")),
    )

    logging.info(
        f"Network:\n"
        f"\t{model.n_channels} input channels\n"
        f"\t{model.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling'
    )

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f"Model loaded from {args.load}")

    cudnn.benchmark = True  # cudnn Autotuner
    cudnn.enabled = True  # look for optimal algorithms

    try:
        train_net(
            model=model,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            dev=device,
            val_percent=args.val / 100,
            image_size=args.img_size,
            val_int=args.val_int,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
