import torch
import torch.nn as nn
import torch.nn.functional as F


class GRURecModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 1,
        padding_idx: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Item embedding
        self.item_embedding = nn.Embedding(
            num_embeddings=num_items,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx
        )

        # GRU encoder
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Projection to item embedding space
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        """
        sequences: Tensor of shape [batch_size, seq_len]
        returns: logits of shape [batch_size, num_items]
        """

        # [B, T] -> [B, T, D]
        embedded = self.item_embedding(sequences)

        # GRU encoding
        # output: [B, T, H], hidden: [num_layers, B, H]
        _, hidden = self.gru(embedded)

        # last layer hidden state
        user_state = hidden[-1]  # [B, H]

        # project to embedding space
        user_embedding = self.output_projection(user_state)  # [B, D]

        # scoring: dot product with all item embeddings
        item_embeddings = self.item_embedding.weight  # [num_items, D]

        logits = torch.matmul(user_embedding, item_embeddings.T)  # [B, num_items]

        return logits
