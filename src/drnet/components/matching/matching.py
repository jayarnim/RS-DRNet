import torch
import torch.nn as nn
from .aggregation.builder import aggregation_builder
from ...functions.generator import fc_block


class NeuralCollaborativeFilteringLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: list,
        dropout: float,
        agg: str,
    ):
        super().__init__()

        # global attr
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.agg = agg

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_emb: torch.Tensor, 
        item_emb: torch.Tensor,
    ):
        X = self.aggregation(user_emb, item_emb)
        return self.mlp(X)

    def _set_up_components(self):
        self._create_layers()

    def _create_layers(self):
        kwargs = dict(
            name=self.agg,
        )
        self.aggregation = aggregation_builder(**kwargs)

        kwargs = dict(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )
        components = list(fc_block(**kwargs))
        self.mlp = nn.Sequential(*components)