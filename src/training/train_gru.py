from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SequenceDataset
from src.models.gru_rec import GRURecModel


PROCESSED_DATA_DIR = Path("data/processed")


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    criterion,
    device,
    grad_clip: float = 5.0
):
    model.train()
    total_loss = 0.0

    for sequences, targets in tqdm(dataloader, desc="Training"):
        sequences = sequences.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        logits = model(sequences)
        loss = criterion(logits, targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    k: int = 20
):
    model.eval()
    hits = 0
    total = 0

    for sequences, targets in tqdm(dataloader, desc="Evaluating"):
        sequences = sequences.to(device)
        targets = targets.to(device)

        logits = model(sequences)

        topk = torch.topk(logits, k=k, dim=1).indices
        hits += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)

    return hits / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SequenceDataset(
        PROCESSED_DATA_DIR / "train_sequences.npy",
        PROCESSED_DATA_DIR / "targets.npy"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )

    num_items = int(train_dataset.sequences.max()) + 1

    model = GRURecModel(
        num_items=num_items,
        embedding_dim=64,
        hidden_dim=128
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 10

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        recall = evaluate(
            model,
            train_loader,
            device,
            k=20
        )

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Recall@20: {recall:.4f}"
        )

    torch.save(
        model.state_dict(),
        "gru_recommender.pt"
    )


if __name__ == "__main__":
    main()
