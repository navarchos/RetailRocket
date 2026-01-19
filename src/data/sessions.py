from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


PROCESSED_DATA_DIR = Path("data/processed")


def load_events() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_DIR / "events.parquet")


def build_user_sequences(
    df: pd.DataFrame,
    min_seq_len: int = 3,
    max_seq_len: int = 20
) -> List[Tuple[List[int], int]]:
    sequences = []

    grouped = df.groupby("user_idx")

    for user_id, user_df in tqdm(grouped, desc="Building sequences"):
        items = user_df["item_idx"].tolist()

        if len(items) < min_seq_len:
            continue

        # last-item holdout
        train_items = items[:-1]
        target_item = items[-1]

    
        if len(train_items) > max_seq_len:
            train_items = train_items[-max_seq_len:]

        sequences.append((train_items, target_item))

    return sequences


def pad_sequences(sequences: List[List[int]], max_len: int) -> np.ndarray:
    padded = np.zeros((len(sequences), max_len), dtype=np.int64)

    for i, seq in enumerate(sequences):
        padded[i, -len(seq):] = seq

    return padded


def save_sequences(
    train_seqs: np.ndarray,
    targets: np.ndarray
):
    np.save(PROCESSED_DATA_DIR / "train_sequences.npy", train_seqs)
    np.save(PROCESSED_DATA_DIR / "targets.npy", targets)



def main():
    df = load_events()

    data = build_user_sequences(
        df,
        min_seq_len=3,
        max_seq_len=20
    )

    train_seqs, targets = zip(*data)

    train_seqs = pad_sequences(train_seqs, max_len=20)
    targets = np.array(targets, dtype=np.int64)

    save_sequences(train_seqs, targets)

    print(f"Total sequences: {len(train_seqs)}")


if __name__ == "__main__":
    main()
