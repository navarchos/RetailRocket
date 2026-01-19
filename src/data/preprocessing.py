import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm
print("Скрипт preprocessing.py начал выполнение...")

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def load_events() -> pd.DataFrame:
    events_path = RAW_DATA_DIR / "events.csv"
    df = pd.read_csv(events_path)

    
    df = df[df["event"].isin(["view", "addtocart", "transaction"])]
    df = df.sort_values("timestamp")

    return df


def remap_ids(df: pd.DataFrame):
    user_ids = df["visitorid"].unique()
    item_ids = df["itemid"].unique()

    user2idx = {u: i + 1 for i, u in enumerate(user_ids)}
    item2idx = {i: j + 1 for j, i in enumerate(item_ids)}

    df["user_idx"] = df["visitorid"].map(user2idx)
    df["item_idx"] = df["itemid"].map(item2idx)

    return df, user2idx, item2idx


def save_mappings(user2idx: dict, item2idx: dict):
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    user2idx = {str(k): int(v) for k, v in user2idx.items()}
    item2idx = {str(k): int(v) for k, v in item2idx.items()}

    with open(PROCESSED_DATA_DIR / "user_mapping.json", "w") as f:
        json.dump(user2idx, f)

    with open(PROCESSED_DATA_DIR / "item_mapping.json", "w") as f:
        json.dump(item2idx, f)


def save_events(df: pd.DataFrame):
    cols = [
        "user_idx",
        "item_idx",
        "event",
        "timestamp"
    ]

    df[cols].to_parquet(
        PROCESSED_DATA_DIR / "events.parquet",
        index=False
    )


def main():
    print("Loading events...")
    df = load_events()

    print("Remapping user and item ids...")
    df, user2idx, item2idx = remap_ids(df)

    print("Saving processed data...")
    save_mappings(user2idx, item2idx)
    save_events(df)

    print(f"Users: {len(user2idx)}")
    print(f"Items: {len(item2idx)}")
    print(f"Events: {len(df)}")


if __name__ == "__main__":
    main()
