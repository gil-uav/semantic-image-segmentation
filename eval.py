import torch
from torch.cuda.amp import autocast
from tqdm import tqdm
from pytorch_lightning.metrics.functional import (
    accuracy,
    precision,
    iou,
    recall,
    f1_score,
)


def eval_net(net, loader, device):
    """
    Evaluates network with accuracy, F1, precision, recall and IoU.

    Parameters
    ----------
    net : nn.Module
        Netword to evaluate.
    loader : nn.Dataloader
        Data to evaluate network on.
    device : torch.Device
        CPU or GPU

    Returns
    -------
    {"Accuracy": _accuracy / val_amount, "F1": _f1 / val_amount, "Precision": _precision / val_amount,
    "Recall": _recall / val_amount, "IOU": _iou / val_amount}
    """
    net.eval()  # Turn off training layers
    mask_type = torch.float32 if net.module.n_classes == 1 else torch.long
    val_amount = len(loader)  # the number of batch
    _accuracy = _f1 = _precision = _recall = _iou = 0

    with tqdm(
        total=val_amount, desc="Validating", unit="batch", position=0, leave=True
    ) as p_bar:
        for batch in loader:
            images, masks = batch["image"], batch["mask"]
            images = images.to(device=device, dtype=torch.float32)
            masks = masks.to(device=device, dtype=mask_type)

            # Automatic mixed precision(Hopefully to float16 to utilize tensor cores)
            with autocast(enabled=True):
                with torch.no_grad():  # Disable gradient calculation
                    masks_pred = net(images)

            pred = torch.sigmoid(masks_pred)
            pred = (pred > 0.5).float()

            # Calculate metrics
            num_classes = net.module.n_classes + 1
            _accuracy += accuracy(pred, masks, num_classes=num_classes).item()
            _f1 += f1_score(pred, masks, num_classes=num_classes).item()
            _precision += precision(pred, masks, num_classes=num_classes).item()
            _recall += recall(pred, masks, num_classes=num_classes).item()
            _iou += iou(pred, masks, num_classes=num_classes).item()

            p_bar.set_postfix(
                **{
                    "A": _accuracy / val_amount,
                    "F1": _f1 / val_amount,
                    "P": _precision / val_amount,
                    "R": _recall / val_amount,
                    "IOU": _iou / val_amount,
                }
            )
            p_bar.update()

    net.train()  # Turn on training layers
    return {
        "Accuracy": _accuracy / val_amount,
        "F1": _f1 / val_amount,
        "Precision": _precision / val_amount,
        "Recall": _recall / val_amount,
        "IOU": _iou / val_amount,
    }
