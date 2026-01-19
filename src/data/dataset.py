from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


PROCESSED_DATA_DIR = Path("data/processed")


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences_path: Path,
        targets_path: Path
    ):
        self.sequences = np.load(sequences_path)
        self.targets = np.load(targets_path)

        assert len(self.sequences) == len(self.targets), \
            "Sequences and targets must have same length"

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        seq = torch.from_numpy(self.sequences[idx]).long()
        target = torch.tensor(self.targets[idx]).long()

        return seq, target
