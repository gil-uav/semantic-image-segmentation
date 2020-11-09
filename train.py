import logging
import os
import sys
from argparse import ArgumentParser
from datetime import datetime

from knockknock import discord_sender

import torch
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from torch.backends import cudnn

from unet.unet_model import UNet

load_dotenv(verbose=True)


@discord_sender(webhook_url=os.getenv("DISCORD_WH"))
def main():
    parser = ArgumentParser()

    parser = UNet.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    prod = bool(os.getenv("PROD"))
    logging.getLogger("lightning").setLevel(logging.INFO)

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

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        mode="min",
        patience=3 if not os.getenv("EARLY_STOP") else int(os.getenv("EARLY_STOP")),
        verbose=True,
    )

    run_name = "{}_LR{}_BS{}_IS{}".format(
        datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
        args.lr,
        args.batch_size,
        args.image_size,
    ).replace(".", "_")

    log_folder = (
        "./logs" if not os.getenv("DIR_ROOT_DIR") else os.getenv("DIR_ROOT_DIR")
    )
    if not os.path.isdir(log_folder):
        os.mkdir(log_folder)
    logger = TensorBoardLogger(
        log_folder,
        name=run_name,
        default_hp_metric=False,
    )

    try:
        trainer = Trainer.from_argparse_args(
            args,
            gpus=-1,
            precision=16,
            distributed_backend="ddp",
            logger=logger,
            callbacks=[early_stop_callback],
            accumulate_grad_batches=1.0
            if not os.getenv("ACC_GRAD")
            else int(os.getenv("ACC_GRAD")),
            gradient_clip_val=0.0
            if not os.getenv("GRAD_CLIP")
            else float(os.getenv("GRAD_CLIP")),
            max_epochs=100 if not os.getenv("EPOCHS") else int(os.getenv("EPOCHS")),
            val_check_interval=100
            if not os.getenv("VAL_INT_PER")
            else float(os.getenv("VAL_INT_PER")),
            default_root_dir=os.getcwd()
            if not os.getenv("DIR_ROOT_DIR")
            else os.getenv("DIR_ROOT_DIR"),
        )
        trainer.fit(model)
        trainer.test(model)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os.exit(0)


if __name__ == "__main__":
    main()
