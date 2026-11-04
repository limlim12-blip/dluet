import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from dataset import IDataset


class IDataLoader(DataLoader):
    def __init__(
        self,
        data_dir,
        validation_split,
        batch_size,
        num_workers,
        seed=123,
        collate_fn=None,
        shuffle_buffer=10_000,  # CHANGED: new, train only
    ):

        # Initialize training dataset
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

        valid_dataset = IDataset(
            data_dir, validation_split=validation_split, seed=seed, is_valid=True
        )
        valid_dataset = ShuffleDataset(valid_dataset, 2048 * 32)
        self.valid_loader = DataLoader(
            dataset=valid_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=num_workers,
        )

    def split_validation(self):
        return self.valid_loader


class ShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size):
        super().__init__()
        self.dataset = dataset
        self.buffer_size = buffer_size

    def __iter__(self):
        shufbuf = []
        try:
            dataset_iter = iter(self.dataset)
            for i in range(self.buffer_size):
                shufbuf.append(next(dataset_iter))
        except:
            self.buffer_size = len(shufbuf)

        try:
            while True:
                try:
                    item = next(dataset_iter)
                    evict_idx = random.randint(0, self.buffer_size - 1)
                    yield shufbuf[evict_idx]
                    shufbuf[evict_idx] = item
                except StopIteration:
                    break
            while len(shufbuf) > 0:
                yield shufbuf.pop()
        except GeneratorExit:
            pass


def collate_fn(batch):
    # list of [chan,size]
    features, targets = zip(*batch)
    # list of [size,chan]
    processed_features = [f.transpose(0, 1) for f in features]
    # batch size chan
    padded_features = pad_sequence(
        processed_features, batch_first=True, padding_value=0.0
    )

    # [batch chan size]
    padded_features = padded_features.permute(0, 2, 1)
    targets = torch.stack(targets, dim=0)

    mask = padded_features.abs().sum(dim=1) > 0
    return padded_features, targets, mask
