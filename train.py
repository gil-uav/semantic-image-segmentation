import logging
import os
import sys
from argparse import ArgumentParser

import torch
from dotenv import load_dotenv
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateLogger
from torch.backends import cudnn

from unet.unet_model import UNet

load_dotenv(verbose=True)


def main():
    parser = ArgumentParser()

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

    lr_monitor = LearningRateLogger(logging_interval="step")
    early_stopping = EarlyStopping(
        "val_loss",
        patience=5 if not os.getenv("EARLY_STOP") else int(os.getenv("EARLY_STOP")),
        verbose=True,
    )

    try:
        trainer = Trainer.from_argparse_args(
            args,
            gpus=-1,
            precision=16,
            distributed_backend="ddp",
            callbacks=[lr_monitor],
            early_stop_callback=early_stopping,
            accumulate_grad_batches=1
            if not os.getenv("ACC_GRAD")
            else int(os.getenv("ACC_GRAD")),
            gradient_clip_val=0.0
            if not os.getenv("GRAD_CLIP")
            else float(os.getenv("GRAD_CLIP")),
            max_epochs=1000 if not os.getenv("EPOCHS") else os.getenv("EPOCHS"),
        )
        trainer.fit(model)
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "INTERRUPTED.pth")
        logging.info("Saved interrupt")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


if __name__ == "__main__":
    main()
