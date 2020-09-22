import argparse
import logging
import os
import sys

import torch
import torch.nn as nn
from dotenv import load_dotenv
from torch import optim
from torch.backends import cudnn
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from eval import eval_net
from unet.unet_model import UNet
from utils.dataset import OrhogonalPhotoDataset

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
):
    """
    Trains the network with the custom OrthogonalDataset. Remember to set mapping and classes in the dataset.

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
    """
    dataset = OrhogonalPhotoDataset(dir_img, dir_mask)
    val_amount = int(len(dataset) * val_percent)
    train_amount = len(dataset) - val_amount

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

    global_step = 0

    logging.info(
        f"""Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {train_amount}
        Validation size: {val_amount}
        Checkpoints:     {save_cp}
        Device:          {dev.type}
    """
    )

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
    )
    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min" if model.n_classes > 1 else "max", patience=8
    )

    # Loss function
    if model.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    model = nn.DataParallel(model)

    # Training
    for epoch in range(epochs):

        epoch_loss = 0
        model.train()

        # Progress bar
        with tqdm(
            total=train_amount, desc=f"Epoch {epoch + 1}/{epochs}", unit="img"
        ) as p_bar:

            # Mini-batch runs
            for batch in train_loader:
                images = batch["image"]
                masks = batch["mask"]
                assert images.shape[1] == model.module.n_channels, (
                    f"Network has been defined with {model.module.n_channels} input channels, "
                    f"but loaded images have {images.shape[1]} channels. Please check that "
                    "the images are loaded correctly."
                )

                # Same as optimizer.zero_grad() but more efficient.
                for param in model.parameters():
                    param.grad = None

                # To device
                images = images.to(device=dev, dtype=torch.float32)
                mask_type = torch.float32 if model.module.n_classes == 1 else torch.long
                masks = masks.to(device=dev, dtype=mask_type, non_blocking=True)

                # Automatic mixed precision(Hopefully to float16 to utilize tensor cores)
                with autocast():
                    masks_pred = model(images)  # Forward pass
                    loss = criterion(masks_pred, masks)  # Compute loss function

                epoch_loss += loss.item()

                if global_step == 0:
                    writer.add_graph(model.module, images)

                writer.add_scalar("Training loss", loss.item(), global_step)
                p_bar.set_postfix(**{"Batch loss": loss.item()})

                loss.backward()  # Backward pass, also Synchronizes GPU
                nn.utils.clip_grad_value_(
                    model.parameters(), 0.1
                )  # To avoid the exploding gradients problem
                optimizer.step()  # Optimizer step

                p_bar.update(images.shape[0])
                global_step += 1

                # Validation
                if global_step % (train_amount // (10 * batch_size)) == 0:

                    logging.info("\nRunning test evaluation.")
                    for tag, value in model.named_parameters():
                        tag = tag.replace(".", "/")
                        writer.add_histogram(
                            "weights/" + tag, value.data.cpu().numpy(), global_step
                        )
                        writer.add_histogram(
                            "grads/" + tag, value.grad.data.cpu().numpy(), global_step
                        )

                    # Run test evaluation
                    test_metrics = eval_net(model, val_loader, dev)
                    scheduler.step(test_metrics["F1"])
                    writer.add_scalar(
                        "Learning rate", optimizer.param_groups[0]["lr"], global_step
                    )

                    for key, val in zip(test_metrics.keys(), test_metrics.values()):
                        logging.info("Validation {}: {}".format(key, val))
                        writer.add_scalar("{}/test".format(key), val, global_step)

                    writer.add_images("Images", images, global_step)
                    if model.module.n_classes == 1:
                        writer.add_images("True masks", masks, global_step)
                        writer.add_images(
                            "Predicted masks",
                            torch.sigmoid(masks_pred) > 0.5,
                            global_step,
                        )
        # Run training evaluation
        logging.info("\nRunning training evaluation.")
        train_metrics = eval_net(model, train_loader, dev)

        for key, val in zip(train_metrics.keys(), train_metrics.values()):
            logging.info("Training {}: {}".format(key, val))
            writer.add_scalar("{}/train".format(key), val, global_step)

        # Save checkpoint
        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info("Created checkpoint directory")
            except OSError:
                pass
            torch.save(model.state_dict(), dir_checkpoint + f"CP_epoch{epoch + 1}.pth")
            logging.info(f"Checkpoint {epoch + 1} saved !")

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(
        description="Train the UNet on images and target masks",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        metavar="E",
        type=int,
        default=5 if not os.getenv("E") else os.getenv("E"),
        help="Number of epochs",
        dest="epochs",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        metavar="B",
        type=int,
        nargs="?",
        default=1 if not os.getenv("BS") else os.getenv("BS"),
        help="Batch size",
        dest="batchsize",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        metavar="LR",
        type=float,
        nargs="?",
        default=0.0001 if not os.getenv("LR") else os.getenv("LR"),
        help="Learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-f",
        "--load",
        dest="load",
        type=str,
        default=False,
        help="Load model from a .pth file",
    )
    parser.add_argument(
        "-v",
        "--validation",
        dest="val",
        type=float,
        default=10.0 if not os.getenv("VAL") else os.getenv("VAL"),
        help="Percent of the data that is used as validation (0-100)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if bool(os.getenv("PROD")):
        logging.info("Training i production mode, disabling all debugging APIs")
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(enabled=False)
        torch.autograd.profiler.emit_nvtx(enabled=False)
    else:
        # TODO: Implement these debuggers
        logging.info("Training i development mode, debugging APIs active.")
        torch.autograd.set_detect_anomaly(True)
        torch.autograd.profiler.profile(
            enabled=True, use_cuda=True, record_shapes=True, profile_memory=True
        )
        torch.autograd.profiler.emit_nvtx(enabled=True, record_shapes=True)

    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device {device}")
    model = UNet(
        n_channels=int(os.getenv("N_CHANNELS")),
        n_classes=int(os.getenv("N_CLASSES")),
        bilinear=bool(os.getenv("BILINEAR")),
    ).cuda()

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

    try:
        model.to(device=device)
        train_net(
            model=model,
            epochs=args.epochs,
            batch_size=args.batchsize,
            lr=args.lr,
            dev=device,
            val_percent=args.val / 100,
        )
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
