import torch
import torch.nn as nn
from components.embedding import *


@register("idx_with_history")
class IDXEmbeddingWithHistory(EmbeddingLayer):
    def __init__(
        self,
        num_items: int,
        embedding_dim: int,
    ):
        super().__init__()

        kwargs = dict(
            num_embeddings=1, 
            embedding_dim=embedding_dim,
        )
        self.query = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=num_items+2, 
            embedding_dim=embedding_dim,
            padding_idx=0,
        )
        self.item_emb = nn.Embedding(**kwargs)
        self.hist_emb = nn.Embedding(**kwargs)

        self.init_embeddings()

    def forward(
        self, 
        item_idx: torch.Tensor,
        hist_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.query.weight
        item_emb = self.item_emb(item_idx)
        hist_emb = self.hist_emb(hist_idx)
        return query, item_emb, hist_emb

    def init_embeddings(self):
        embeddings = [
            self.query,
            self.item_emb,
            self.hist_emb,
        ]

        for emb in embeddings:
            kwargs = dict(
                tensor=emb.weight, 
                mean=0.0, 
                std=0.01,
            )
            nn.init.normal_(**kwargs)