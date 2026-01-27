from pathlib import Path

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
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

    for items, events, targets in tqdm(dataloader, desc="Training"):
        items = items.to(device)
        events = events.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        user_emb = model(items, events, return_user_emb=True)

        loss = sampled_softmax_loss(
            user_emb=user_emb,
            target_ids=targets,
            item_embedding=model.item_embedding
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def sampled_softmax_loss(user_emb, target_ids, item_embedding, temperature=1.0):
    """
    user_emb:     [B, D]
    target_ids:   [B]
    item_embedding: nn.Embedding
    """

    # [B, D]
    target_embs = item_embedding(target_ids)

    # logits: [B, B]
    logits = torch.matmul(user_emb, target_embs.T) / temperature

    labels = torch.arange(len(target_ids), device=logits.device)

    loss = F.cross_entropy(logits, labels)
    return loss



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

    for items, events, targets in tqdm(dataloader, desc="Evaluating"):
        items = items.to(device)
        events = events.to(device)
        targets = targets.to(device)

        logits = model(items, events)

        topk = torch.topk(logits, k=k, dim=1).indices
        hits += (topk == targets.unsqueeze(1)).any(dim=1).sum().item()
        total += targets.size(0)

    return hits / total

@torch.no_grad()
def evaluate_metrics(model, dataloader, device, k: int = 20):
    model.eval()
    total = 0
    recall_hits = 0
    rr_sum = 0.0

    for items, events, targets in dataloader:
        items = items.to(device)
        events = events.to(device)
        targets = targets.to(device)

        logits = model(items, events)

        # [B, K]
        topk = torch.topk(logits, k=k, dim=1).indices

        # [B, K] == [B, 1] -> [B, K]
        matches = topk.eq(targets.unsqueeze(1))

        # Recall@K
        recall_hits += matches.any(dim=1).sum().item()

        # MRR@K
        # ranks: [B], where match exists, else 0
        # argmax returns first True because False=0, True=1
        ranks = torch.argmax(matches.int(), dim=1) + 1

        # mask: target appeared in top-k
        mask = matches.any(dim=1)

        rr_sum += (1.0 / ranks[mask].float()).sum().item()

        total += targets.size(0)

    return {
        "recall": recall_hits / total,
        "mrr": rr_sum / total
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SequenceDataset(
        items_path=PROCESSED_DATA_DIR / "train_items.npy",
        events_path=PROCESSED_DATA_DIR / "train_events.npy",
        targets_path=PROCESSED_DATA_DIR / "train_targets.npy"
    )

    val_dataset = SequenceDataset(
        items_path=PROCESSED_DATA_DIR / "val_items.npy",
        events_path=PROCESSED_DATA_DIR / "val_events.npy",
        targets_path=PROCESSED_DATA_DIR / "val_targets.npy"
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    with open(PROCESSED_DATA_DIR / "item_mapping.json") as f:
        item2idx = json.load(f)

    num_items = len(item2idx) + 1  # + padding idx = 0

    assert train_dataset.items.max() < num_items
    assert train_dataset.targets.max() < num_items

    assert val_dataset.items.max() < num_items
    assert val_dataset.targets.max() < num_items

    model = GRURecModel(
        num_items=num_items,
        num_events=4,     # view / cart / purchase
        item_dim=64,
        event_dim=8,
        hidden_dim=128
    ).to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 15
    assert targets.unique().numel() > len(targets) * 0.9

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device
        )

        # val_recall = evaluate(
        #     model,
        #     val_loader,
        #     device,
        #     k=20
        # )

        metrics = evaluate_metrics(model, val_loader, device, k=20)

        print(
            f"Epoch {epoch:02d} | "
            f"Loss: {train_loss:.4f} | "
            f"Recall@20: {metrics['recall']:.4f} | "
            f"MRR@20: {metrics['mrr']:.4f}"
        )

    torch.save(
        model.state_dict(),
        "gru_recommender.pt"
    )


if __name__ == "__main__":
    main()
