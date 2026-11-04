import ast
import glob
import math
import os
import random

import numpy as np
import orjson
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

IPT_CH, SIZE_CH = 0, 2
IPT_SCALE = math.log1p(10_000)
SIZE_SCALE = math.log1p(1_500)


class IDataset(IterableDataset):
    def __init__(
        self, data_dir, seed=123, validation_split=0.2, is_valid=False, shuffle_buffer=0
    ):
        super().__init__()
        self.data_dir = data_dir
        self.shuffle_buffer = shuffle_buffer
        rng = np.random.default_rng(seed=seed)
        all_files = sorted(
            f
            for f in glob.glob(os.path.join(self.data_dir, "**/*.csv"), recursive=True)
            if "dataset-statistics" not in f
        )
        rng.shuffle(all_files)

        split_idx = int(len(all_files) * validation_split)
        if is_valid:
            self.files = all_files[:2]
        else:
            self.files = all_files[2:]

        self.map = self.load_labels()

    def _process_chunk(self, df):
        ppi = df["PPI"].values
        url_labels = df["APP"].values

        for i in range(len(df)):
            feature_tensor = torch.as_tensor(orjson.loads(ppi[i]), dtype=torch.float32)
            feature_tensor[IPT_CH] = (
                torch.log1p(feature_tensor[IPT_CH].clamp(min=0)) / IPT_SCALE
            )
            feature_tensor[SIZE_CH] = (
                torch.log1p(feature_tensor[SIZE_CH].clamp(min=0)) / SIZE_SCALE
            )

            idx = self.map.get(url_labels[i])
            if idx is None:
                raise KeyError(f"Unknown APP label: {url_labels[i]!r}")
            target_url = torch.tensor(idx, dtype=torch.long)

            yield feature_tensor, target_url

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        worker_id, num_workers = (0, 1) if info is None else (info.id, info.num_workers)

        if len(self.files) >= num_workers:
            work_files = self.files[worker_id::num_workers]
            for file_path in work_files:
                try:
                    for df in pd.read_csv(
                        file_path, chunksize=10000, usecols=["APP", "PPI"]
                    ):
                        yield from self._process_chunk(df)
                except Exception as e:
                    print(f"Could not read {file_path}: {type(e).__name__}: {e}")
        else:
            for file_path in self.files:
                try:
                    for chunk_idx, df in enumerate(
                        pd.read_csv(file_path, chunksize=10000, usecols=["APP", "PPI"])
                    ):
                        if chunk_idx % num_workers == worker_id:
                            yield from self._process_chunk(df)
                except Exception as e:
                    print(f"Could not read {file_path}: {type(e).__name__}: {e}")

    def load_labels(self):
        df = pd.read_csv("./data/cesnet-quic22/dataset-statistics/app.csv")
        self.num_classes = len(df["LABEL"])
        return {label: idx for idx, label in enumerate(sorted(df["LABEL"].to_list()))}


class IDataLoader(DataLoader):
    def __init__(
        self,
        data_dir,
        validation_split,
        batch_size,
        num_workers,
        seed=123,
        collate_fn=None,
        shuffle_buffer=50_000,
    ):

        # Training dataset
        train_dataset = IDataset(
            data_dir,
            validation_split=validation_split,
            seed=seed,
            is_valid=False,
            shuffle_buffer=shuffle_buffer,
        )
        self.num_classes = train_dataset.num_classes
        train_dataset = ShuffleDataset(train_dataset, 2048 * 128)

        super().__init__(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        # Validation dataset (removed heavy ShuffleDataset wrapper to speed up evaluation startup)
        valid_dataset = IDataset(
            data_dir, validation_split=validation_split, seed=seed, is_valid=True
        )
        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    def split_validation(self):
        return self.valid_loader
