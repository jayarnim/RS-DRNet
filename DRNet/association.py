import torch
import torch.nn as nn
from .components.att.model import AttentionMechanism
from .components.embedding.builder import embedding_builder
from .components.matching.builder import matching_fn_builder
from .components.prediction import ProjectionLayer
from . import affection


class Module(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        hidden_dim: list,
        beta: float,
        dropout: float,
        agg: str,
        affection: nn.Module,
        histories: torch.Tensor,
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
            num_users (int):
                total number of users in the dataset, U.
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
            histories (torch.Tensor): 
                historical item interactions for each user, represented as item indices.
                (shape: [U, history_length])
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.dropout = dropout
        self.agg = agg
        self.affection = affection
        self.histories = histories
        self.pred_dim = self.hidden_dim[-1]

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        emb, hist_idx, mask = self.embedding(user_idx, item_idx)
        K = self._K_generator(user_idx, hist_idx)
        user_pooled = self.pooling(emb["query"], K, emb["history"], mask)
        X_pred = self.matching(user_pooled, emb["anchor"])
        return X_pred

    def estimate(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method
        -----

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    def _K_generator(self, user_idx, hist_idx):
        # shape definition
        B, H = hist_idx.shape
        # (B,) -> (B,H) -> (B*H,)
        user_idx_flat = user_idx.unsqueeze(1).expand_as(hist_idx).reshape(-1)
        # (B,H) -> (B*H,)
        hist_idx_flat = hist_idx.reshape(-1)
        # (B*H,D) -> (B,H,D)
        with torch.no_grad():
            self.affection.eval()
            K = self.affection(user_idx_flat, hist_idx_flat).view(B, H, -1)
        return K

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        kwargs = dict(
            history=True,
            num_items=self.num_items,
            embedding_dim=self.embedding_dim,
            histories=self.histories,
        )
        self.embedding = embedding_builder(**kwargs)

        kwargs = dict(
            beta=self.beta,
        )
        self.pooling = AttentionMechanism(**kwargs)

        kwargs = dict(
            input_dim=(
                self.embedding_dim*2
                if self.agg=="cat"
                else self.embedding_dim
            ),
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            agg=self.agg,
        )
        self.matching = matching_fn_builder(**kwargs)

        kwargs = dict(
            dim=self.hidden_dim[-1],
        )
        self.prediction = ProjectionLayer(**kwargs)

    def _assert_arg_error(self):
        CONDITION = (self.affection.hidden_dim[-1] == self.embedding_dim)
        ERROR_MESSAGE = f"last unit of matching function @ affection network must match input size: {self.embedding_dim}"
        assert CONDITION, ERROR_MESSAGE