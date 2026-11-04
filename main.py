import argparse
import collections

import numpy as np
import torch
from torch import optim

import wandb
from data_loader import IDataLoader, collate_fn
from model import ArcFace, FocalLoss, accuracy, top_k_acc
from parse_config import ConfigParser
from train_def import Trainer
from utils import prepare_device

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
torch.set_num_threads(torch.get_num_threads())
torch.backends.cudnn.deterministic = False
np.random.seed(SEED)
torch.set_float32_matmul_precision("high")


def main(config):
    logger = config.get_logger("train")
    logger.info(f"configuration: {config.config}")

    data_loader = IDataLoader(seed=SEED, **config["data_loader"], collate_fn=collate_fn)
    valid_data_loader = data_loader.split_validation()
    model = ArcFace(**config["arch"], num_classes=data_loader.num_classes)
    logger.info(model)
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    criterion = FocalLoss(gamma=2.0)
    metrics = [accuracy, top_k_acc]

    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Biases and 1D normalization weights (GroupNorm, BatchNorm) have ndim == 1
        if param.ndim <= 1 or "bias" in name or "norm" in name:
            no_decay_params.append(param)
        else:
            # 2D/3D Convolution and Linear weights get decayed
            decay_params.append(param)

    optim_groups = [
        {"params": decay_params, "weight_decay": config["optimizer"]["weight_decay"]},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(
        optim_groups,
        lr=config["optimizer"]["lr"],
        fused=config["optimizer"].get("fused", True),
    )
    lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, **config["lr_scheduler"]
    )

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )
    trainer.train()


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default="./config.json",
        type=str,
        help="config file path (default: ./config.json)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    options = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;lr"),
        CustomArgs(["--bs", "--batch_size"], type=int, target="data_loader;batch_size"),
    ]
    args.parse_args = lambda: args.parse_known_args()[0]
    config = ConfigParser.from_args(args, options)
    main(config)
