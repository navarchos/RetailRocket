import torch
import torch.nn as nn


class GRURecModel(nn.Module):
    def __init__(
        self,
        num_items: int,
        num_events: int,
        item_dim: int,
        event_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 1,
        padding_idx: int = 0,
        dropout: float = 0.1
    ):
        super().__init__()

        self.num_items = num_items
        self.item_dim = item_dim
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.item_embedding = nn.Embedding(
            num_items,
            item_dim,
            padding_idx=padding_idx
        )

        self.event_embedding = nn.Embedding(
            num_events,
            event_dim,
            padding_idx=padding_idx
        )

        # GRU
        self.gru = nn.GRU(
            input_size=item_dim + event_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Project hidden state → item embedding space
        self.output_projection = nn.Linear(hidden_dim, item_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def forward(self, item_seq, event_seq, return_user_emb=False) -> torch.Tensor:
        """
        item_seq:  [B, T]
        event_seq: [B, T]
        returns:   logits [B, num_items]
        """

        item_emb = self.item_embedding(item_seq)     # [B, T, item_dim]
        event_emb = self.event_embedding(event_seq)  # [B, T, event_dim]

        x = torch.cat([item_emb, event_emb], dim=-1)

        _, hidden = self.gru(x)
        user_state = hidden[-1]                       # [B, hidden_dim]

        user_emb = self.output_projection(user_state) # [B, item_dim]

        if return_user_emb:
            return user_emb
            
        item_embs = self.item_embedding.weight        # [num_items, item_dim]
        logits = torch.matmul(user_emb, item_embs.T)  # [B, num_items]

        return logits
