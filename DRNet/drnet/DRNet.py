import torch
import torch.nn as nn
from . import affection, association


class Module(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        n_factors: int,
        hidden: list,
        dropout: float,
        user_hist: torch.Tensor, 
    ):
        """
        Dual relations network for collaborative filtering (Ji et al., 2020)
        -----
        Implements the base structure of Dual Relations Network (DRNet),
        MLP & id embedding based latent factor model,
        applying attention mechanism to aggregate histories,
        combining an affection and an association
        to learn user-item and item-item interactions.

        Args:
            n_users (int):
                total number of users in the dataset, U.
            n_items (int):
                total number of items in the dataset, I.
            n_factors (int):
                dimensionality of user and item latent representation vectors, K.
            hidden (int):
                layer dimensions for the MLP-based matching function.
                (e.g., [64, 32, 16, 8])
            dropout (float):
                dropout rate applied to MLP layers for regularization.
            user_hist (torch.Tensor): 
                historical item interactions for each user, represented as item indices.
                (shape: [U, history_length])
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.n_users = n_users
        self.n_items = n_items
        self.n_factors = n_factors
        self.hidden = hidden
        self.dropout = dropout
        self.register_buffer(
            name="user_hist", 
            tensor=user_hist,
        )

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Training Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])
        
        Returns:
            logit (torch.Tensor): (u,i) pair interaction logit (shape: [B,])
        """
        return self.score(user_idx, item_idx)

    @torch.no_grad()
    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        Evaluation Method

        Args:
            user_idx (torch.Tensor): target user idx (shape: [B,])
            item_idx (torch.Tensor): target item idx (shape: [B,])

        Returns:
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        logit = self.score(user_idx, item_idx)
        prob = torch.sigmoid(logit)
        return prob

    def score(self, user_idx, item_idx):
        pred_vector = self.ensemble(user_idx, item_idx)
        logit = self.logit_layer(pred_vector).squeeze(-1)
        return logit

    def ensemble(self, user_idx, item_idx):
        # modules
        pred_vector_affection = self.affection.ncf(user_idx, item_idx)
        pred_vector_association = self.association.ncf(user_idx, item_idx)

        # agg
        kwargs = dict(
            tensors=(pred_vector_affection, pred_vector_association), 
            dim=-1,
        )
        pred_vector = torch.cat(**kwargs)

        return pred_vector

    def _set_up_components(self):
        self._create_modules()
        self._create_layers()

    def _create_modules(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden=self.hidden,
            dropout=self.dropout,
        )
        self.affection = affection.Module(**kwargs)

        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors,
            hidden=self.hidden,
            dropout=self.dropout,
            user_hist=self.user_hist,
        )
        self.association = association.Module(**kwargs)

    def _create_layers(self):
        kwargs = dict(
            in_features=self.n_factors//2 + self.hidden[-1],
            out_features=1,
        )
        self.logit_layer = nn.Linear(**kwargs)