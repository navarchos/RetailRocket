from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    def __init__(
        self,
        items_path: Path,
        events_path: Path,
        targets_path: Path
    ):
        self.items = np.load(items_path)
        self.events = np.load(events_path)
        self.targets = np.load(targets_path)

        assert len(self.items) == len(self.events) == len(self.targets), \
            "Items, events and targets must have same length"

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx):
        items = torch.from_numpy(self.items[idx]).long()
        events = torch.from_numpy(self.events[idx]).long()
        target = torch.tensor(self.targets[idx]).long()

        return items, events, target


