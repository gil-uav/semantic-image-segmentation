import logging
import os
import sys
from argparse import ArgumentParser

import torch
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from torch.backends import cudnn

from unet.unet_model import UNet

load_dotenv(verbose=True)


def add_program_specific_args(parent_parser):
    parser = ArgumentParser(
        parents=[parent_parser],
        add_help=False,
        description="Train the UNet on images and target masks",
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
    parser.add_argument(
        "-cdp",
        "--checkpoint-data-path",
        metavar="CDP",
        dest="checkpoints_path",
        type=str,
        default="checkpoints/"
        if not os.getenv("DIR_CHECKPOINTS")
        else os.getenv("DIR_CHECKPOINTS"),
        help="Load model from a .pth file",
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

    return parser


if __name__ == "__main__":
    logging.getLogger("lightning").setLevel(logging.ERROR)

    parser = ArgumentParser()

    parser = add_program_specific_args(parser)
    parser = UNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    prod = bool(os.getenv("PROD"))
    if prod:
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

    model = UNet(**vars(args))

    logging.info(
        f"Network:\n"
        f"\t{model.hparams.n_channels} input channels\n"
        f"\t{model.hparams.n_classes} output channels (classes)\n"
        f'\t{"Bilinear" if model.hparams.bilinear else "Transposed conv"} upscaling'
    )

    cudnn.benchmark = True  # cudnn Autotuner
    cudnn.enabled = True  # look for optimal algorithms

    try:
        trainer = Trainer.from_argparse_args(
            args,
            gpus=-1,
            precision=16,
            distributed_backend="ddp",
        )
        trainer.fit(model)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
