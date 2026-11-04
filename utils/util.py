import json
from collections import OrderedDict
from itertools import repeat
from pathlib import Path

import pandas as pd
import torch
import wandb


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def read_json(fname):
    fname = Path(fname)
    with fname.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content, fname):
    fname = Path(fname)
    with fname.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print(n_gpu, n_gpu_use)
        print(
            "Warning: There's no GPU available on this machine,"
            "training will be performed on CPU."
        )
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(
            f"Warning: The number of GPU's configured to use is {n_gpu_use}, but only {n_gpu} are "
            "available on this machine."
        )
        n_gpu_use = n_gpu
    device = torch.device("cuda:0" if n_gpu_use > 0 else "cpu")
    list_ids = list(range(n_gpu_use))
    return device, list_ids


class MetricTracker:
    def __init__(self, *keys, writer=None, section="train", use_wandb=False):
        self.writer = writer
        self._data = pd.DataFrame(
            index=keys, columns=["total", "counts", "average"], dtype=float
        )
        self.reset()
        self.use_wandb = use_wandb
        self.section = section

    def reset(self):
        self._data.loc[:, :] = 0.0

    def update(self, key, value, n=1, step=None):
        if self.writer is not None:
            self.writer.add_scalar(
                key,
                value,
            )

        if self.use_wandb and self.section == "train":
            log_dict = {f"{self.section}_{key}": value}
            if step is not None:
                log_dict["global_train_step"] = step
            wandb.log(log_dict)

        self._data.loc[key, "total"] += float(value * n)
        self._data.loc[key, "counts"] += int(n)
        self._data.loc[key, "average"] = (
            self._data.loc[key, "total"] / self._data.loc[key, "counts"]
        )

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)
