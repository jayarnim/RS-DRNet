import torch
import torch.nn as nn
from .affection import AffectionNetworks
from .association import AssociationNetworks
from components.base import BaseModel
from .layers.fusion import ConcatenationLayer
from .layers.prediction import ProjectionLayer


class DualRelationNetworks(BaseModel):
    def __init__(
        self,
        affection: AffectionNetworks, 
        association: AssociationNetworks,
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
        super().__init__(locals())

        # ENSEMBLE MODULES ==========
        self.affection = affection
        self.association = association
        self.pred_dim = affection.pred_dim + association.pred_dim

        # FUSION ==========
        self.fusion = ConcatenationLayer()

        # PREDICTION ==========
        self.prediction = ProjectionLayer(
            dim=self.pred_dim,
        )

    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        # ENSEMBLE LEARNING ==========
        args = (
            self.affection(user_idx, item_idx),
            self.association(user_idx, item_idx),
        )
        # ENSEMBLE AGGREGATION ==========
        X_pred = self.fusion(*args)
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
