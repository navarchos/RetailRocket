from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


PROCESSED_DATA_DIR = Path("data/processed")
SESSION_GAP_MS = 30 * 60 * 1000  # 30мин


def load_events() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DATA_DIR / "events.parquet")


def split_into_sessions(
    timestamps: list[int],
    items: list[int],
    session_gap_ms: int = SESSION_GAP_MS
):
    sessions = []
    current_session = [items[0]]

    for i in range(1, len(items)):
        if timestamps[i] - timestamps[i - 1] > session_gap_ms:
            if len(current_session) > 1:
                sessions.append(current_session)
            current_session = []

        current_session.append(items[i])

    if len(current_session) > 1:
        sessions.append(current_session)

    return sessions


def build_user_sequences(
    df: pd.DataFrame,
    min_seq_len: int = 2,
    max_seq_len: int = 20
):
    train, val = [], []

    grouped = df.groupby("user_idx")

    for user_id, user_df in tqdm(grouped, desc="Building sessions"):
        user_df = user_df.sort_values("timestamp")

        items = user_df["item_idx"].tolist()
        events = user_df["event_idx"].tolist()
        timestamps = user_df["timestamp"].tolist()

        if len(items) < min_seq_len + 1:
            continue

        sessions = split_into_sessions(
            timestamps,
            list(zip(items, events))
        )

        for session in sessions:
            if len(session) < min_seq_len + 1:
                continue

            train_pairs = session[:-2]
            train_target = session[-2][0]

            val_pairs = session[:-1]
            val_target = session[-1][0]

            if len(train_pairs) >= min_seq_len:
                train_pairs = train_pairs[-max_seq_len:]
                train.append((train_pairs, train_target))

            val_pairs = val_pairs[-max_seq_len:]
            val.append((val_pairs, val_target))


    return train, val


def pad_sequences(pairs, max_len):
    items = np.zeros((len(pairs), max_len), dtype=np.int64)
    events = np.zeros((len(pairs), max_len), dtype=np.int64)

    for i, seq in enumerate(pairs):
        seq = seq[-max_len:]
        item_seq, event_seq = zip(*seq)

        items[i, -len(seq):] = item_seq
        events[i, -len(seq):] = event_seq

    return items, events


def main():
    df = load_events()

    train_data, val_data = build_user_sequences(
        df,
        min_seq_len=3,
        max_seq_len=20
    )

    # train 
    train_seqs, train_targets = zip(*train_data)
    train_items, train_events = pad_sequences(train_seqs, max_len=20)
    train_targets = np.array(train_targets, dtype=np.int64)

    # val
    val_seqs, val_targets = zip(*val_data)
    val_items, val_events = pad_sequences(val_seqs, max_len=20)
    val_targets = np.array(val_targets, dtype=np.int64)

    # save 
    np.save(PROCESSED_DATA_DIR / "train_items.npy", train_items)
    np.save(PROCESSED_DATA_DIR / "train_events.npy", train_events)
    np.save(PROCESSED_DATA_DIR / "train_targets.npy", train_targets)

    np.save(PROCESSED_DATA_DIR / "val_items.npy", val_items)
    np.save(PROCESSED_DATA_DIR / "val_events.npy", val_events)
    np.save(PROCESSED_DATA_DIR / "val_targets.npy", val_targets)

    print(f"Train sequences: {len(train_seqs)}")
    print(f"Val sequences:   {len(val_seqs)}")


if __name__ == "__main__":
    main()
