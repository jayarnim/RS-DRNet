import torch
import torch.nn as nn
from . import affection, association
from .components.fusion import FusionLayer
from .components.prediction import ProjectionLayer


class Module(nn.Module):
    def __init__(
        self,
        affection: nn.Module, 
        association: nn.Module,
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
            affection (nn.Module)
            association (nn.Moudle)
        """
        super().__init__()

        # attr dictionary for load
        self.init_args = locals().copy()
        del self.init_args["self"]
        del self.init_args["__class__"]

        # global attr
        self.affection = affection
        self.association = association
        self.pred_dim = affection.pred_dim + association.pred_dim

        # generate layers
        self._set_up_components()

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        args = (
            self.affection(user_idx, item_idx),
            self.association(user_idx, item_idx),
        )
        X_pred = self.fusion(*args)
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
            prob (torch.Tensor): (u,i) pair interaction probability (shape: [B,])
        """
        X_pred = self.forward(user_idx, item_idx)
        logit = self.prediction(X_pred)
        return logit

    def _set_up_components(self):
        self._create_components()

    def _create_components(self):
        self.fusion = FusionLayer()

        kwargs = dict(
            dim=self.pred_dim,
        )
        self.prediction = ProjectionLayer(**kwargs)