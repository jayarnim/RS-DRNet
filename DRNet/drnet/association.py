import torch
import torch.nn as nn
from ..attn.model import AttentionMechanism
from . import affection


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
        Implements the base structure of association,
        MLP & id embedding based collaborative filtering model,
        applying attention mechanism to aggregate histories,
        submodule of Dual Relations Network (DRNet)
        to learn item-item interactions.

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

    def score(
        self,
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        pred_vector = self.ncf(user_idx, item_idx)
        logit = self.pred_layer(pred_vector).squeeze(-1)
        return logit

    def ncf(self, user_idx, item_idx):
        user_embed_slice_hist = self.user_hist_embed_generator(user_idx, item_idx)
        item_embed_slice_id = self.item_embed_target(item_idx)

        kwargs = dict(
            tensors=(user_embed_slice_hist, item_embed_slice_id), 
            dim=-1,
        )
        concat = torch.cat(**kwargs)
        pred_vector = self.matching_fn(concat)

        return pred_vector

    def user_hist_embed_generator(self, user_idx, item_idx):
        kwargs = dict(
            target_hist=self.user_hist, 
            target_idx=user_idx, 
            counterpart_padding_idx=self.n_items,
        )
        refer_idx = self._hist_idx_slicer(**kwargs)

        kwargs = dict(
            hist_idx_slice=refer_idx,
            counterpart_idx=item_idx, 
            counterpart_padding_idx=self.n_items,
        )
        mask = self._mask_generator(**kwargs)
        
        kwargs = dict(
            Q=self.user_embed_global.unsqueeze(0),
            K=self.refer_k_generator(user_idx, refer_idx),
            V=self.item_embed_hist(refer_idx),
            mask=mask,
        )
        context = self.attn(**kwargs)

        return context

    def refer_k_generator(self, user_idx, refer_idx):
        B, H = refer_idx.size()
        # (B,) -> (B,H)
        user_idx_exp = user_idx.unsqueeze(1).expand_as(refer_idx)
        # (B,H) -> (B*H,)
        user_idx_flat = user_idx_exp.reshape(-1)
        # (B,H) -> (B*H,)
        refer_idx_flat = refer_idx.reshape(-1)
        # (B*H,D)
        refer_k_flat = self.affection.ncf(user_idx_flat, refer_idx_flat)
        # (B*H,D) -> (B,H,D)
        refer_k = refer_k_flat.view(B, H, -1)
        return refer_k

    def _mask_generator(self, hist_idx_slice, counterpart_idx, counterpart_padding_idx):
        # mask to current target item from history
        marking_counterpart_idx = hist_idx_slice == counterpart_idx.unsqueeze(1)
        # mask to padding
        marking_padding_idx = hist_idx_slice == counterpart_padding_idx
        # final mask
        mask = ~(marking_counterpart_idx | marking_padding_idx)
        return mask

    def _hist_idx_slicer(self, target_hist, target_idx, counterpart_padding_idx):
        # target hist slice
        hist_idx_slice = target_hist[target_idx]
        # calculate max hist in batch
        lengths = (hist_idx_slice != counterpart_padding_idx).sum(dim=1)
        max_len = lengths.max().item()
        # drop padding values
        hist_idx_slice_trunc = hist_idx_slice[:, :max_len]
        return hist_idx_slice_trunc

    def _set_up_components(self):
        self._create_modules()
        self._create_embeddings()
        self._init_embeddings()
        self._create_layers()

    def _create_modules(self):
        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors*2,
            hidden=[unit * 2 for unit in self.hidden],
            dropout=self.dropout,
        )
        self.affection = affection.Module(**kwargs)

    def _create_embeddings(self):
        self.user_embed_global = nn.Parameter(torch.randn(self.n_factors))

        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.item_embed_target = nn.Embedding(**kwargs)
        self.item_embed_hist = nn.Embedding(**kwargs)

    def _init_embeddings(self):
        nn.init.normal_(self.user_embed_global, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embed_target.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.item_embed_hist.weight, mean=0.0, std=0.01)

    def _create_layers(self):
        self.attn = AttentionMechanism()

        components = list(self._yield_layers(self.hidden))
        self.matching_fn = nn.Sequential(*components)

        kwargs = dict(
            in_features=self.hidden[-1],
            out_features=1,
        )
        self.pred_layer = nn.Linear(**kwargs)

    def _yield_layers(self, hidden):
        idx = 1
        while idx < len(hidden):
            yield nn.Linear(hidden[idx-1], hidden[idx])
            yield nn.LayerNorm(hidden[idx])
            yield nn.ReLU()
            yield nn.Dropout(self.dropout)
            idx += 1

    def _assert_arg_error(self):
        CONDITION = (self.hidden[-1] == self.n_factors//2)
        ERROR_MESSAGE = f"Last MLP layer must match input size: {self.n_factors//2}"
        assert CONDITION, ERROR_MESSAGE