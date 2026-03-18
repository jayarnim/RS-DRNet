import torch
from components.histories import Histories
from components.base import BaseModel
from .affection import AffectionNetworks
from .layers.embedding import build as build_embedding_layer
from .layers.att import AttentionMechanism
from .layers.matching import build as build_matching_layer
from .layers.prediction import ProjectionLayer


class AssociationNetworks(BaseModel):
    def __init__(
        self,
        affection: AffectionNetworks,
        histories: Histories,
        num_items: int,
        embedding_dim: int,
        hidden_dim: list,
        beta: float,
        dropout: float,
    ):
        """
        Dual relations network for collaborative filtering (Ji et al., 2020)
        -----
        Implements the base structure of association,
        MLP & id embedding based collaborative filtering model,
        applying attention mechanism to aggregate histories,
        submodule of Dual Relations Network (DRNet)
        to learn item-item interactions.

        Args:
            num_items (int):
                total number of items in the dataset, I.
            embedding_dim (int):
                dimensionality of user and item latent representation vectors, K.
            hidden_dim (int):
                layer dimensions for the MLP-based matching function.
                (e.g., [64, 32, 16, 8])
            beta (float):
                smoothing factor for normalization @ simplex.
                (range: (0,1])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
            affection (nn.Module):
                affection module to generate key @ attention mechanism.
            histories (Histories): 
                historical item interactions for each user, represented as item indices.
                (shape: [U, history_length])
        """
        super().__init__(locals())

        # AFFECTION PRED DIM == EMBEDDING DIM
        CONDITION = (affection.pred_dim == embedding_dim)
        ERROR_MESSAGE = f"last unit of matching function @ affection network must match input size: {embedding_dim}"
        assert CONDITION, ERROR_MESSAGE

        self.pred_dim = hidden_dim[-1]

        # PRETRAINED MODULE TO CALC KEY ==========
        self.affection = affection

        # HISTORY IDX VIEWER ==========
        self.histories = histories

        # IDX EMBEDDING ==========
        self.embedding = build_embedding_layer(
            name="idx_with_history",
            num_items=num_items,
            embedding_dim=embedding_dim,
        )

        # HISTORY POOLING ==========
        self.pooling = AttentionMechanism(
            score="dot",
            dim=embedding_dim,
            beta=beta,
            dropout=dropout,
        )

        # MATCHING FUNCTION LEARNING ==========
        self.matching = build_matching_layer(
            name="ncf",
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=self.pred_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # SEARCH HISTORY IDX ==========
        hist_idx, mask = self.histories(user_idx, item_idx)
        # IDX EMBEDDING ==========
        query, item_emb, hist_emb = self.embedding(item_idx, hist_idx)
        # HISTORY POOLING ==========
        K = self.refer_k(user_idx, hist_idx)
        user_pooled = self.pooling(query, K, hist_emb, mask)
        # MATCHING FUNCTION LEARNING ==========
        X_pred = self.matching(user_pooled, item_emb)
        # PRED VEC ==========
        return X_pred

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        # INTERACTION MODELING ==========
        X_pred = self.forward(user_idx, item_idx)
        # PREDICTION ==========
        logit = self.prediction(X_pred)
        return logit

    def refer_k(self, user_idx, hist_idx):
        # shape definition
        B, H = hist_idx.shape
        # (B,) -> (B,1) -> (B,H) -> (B*H,)
        user_idx_flat = user_idx.unsqueeze(1).expand_as(hist_idx).reshape(-1)
        # (B,H) -> (B*H,)
        hist_idx_flat = hist_idx.reshape(-1)
        # (B*H,D) -> (B,H,D)
        with torch.no_grad():
            self.affection.eval()
            K = self.affection(user_idx_flat, hist_idx_flat).view(B,H,-1)
        return K
