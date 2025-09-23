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
        super(Module, self).__init__()
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
        self._init_layers()


    def forward(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        return self.score(user_idx, item_idx)

    def predict(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        """
        user_idx: (B,)
        item_idx: (B,)
        """
        with torch.no_grad():
            logit = self.score(user_idx, item_idx)
            pred = torch.sigmoid(logit)
        return pred

    def score(
        self,
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        pred_vector = self.gmf(user_idx, item_idx)
        logit = self.logit_layer(pred_vector).squeeze(-1)
        return logit

    def gmf(
        self, 
        user_idx: torch.Tensor, 
        item_idx: torch.Tensor,
    ):
        global_item_vector = self.global_item_vector(user_idx, item_idx)
        item_slice = self.embed_target(item_idx)
        pred_vector = global_item_vector * item_slice
        return pred_vector

    def global_item_vector(self, user_idx, item_idx):
        kwargs = dict(
            target_idx=user_idx, 
            target_hist=self.user_hist, 
            counterpart_padding_value=self.n_items,
        )
        refer_idx = self._hist_slicer(**kwargs)

        kwargs = dict(
            counterpart_idx=item_idx, 
            target_hist_slice=refer_idx,
            counterpart_padding_value=self.n_items,
        )
        mask = self._mask_generator(**kwargs)

        query = self.embed_global.unsqueeze(0)
        refer_k = self.refer_k_calculator(user_idx, refer_idx)
        refer_v = self.embed_hist(refer_idx)
        
        kwargs = dict(
            Q=query,
            K=refer_k,
            V=refer_v,
            mask=mask,
        )
        context = self.attn(**kwargs)

        return context

    def refer_k_calculator(self, user_idx, refer_idx):
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

    def _mask_generator(self, counterpart_idx, target_hist_slice, counterpart_padding_value):
        # mask to current target item from history
        mask_counterpart = target_hist_slice == counterpart_idx.unsqueeze(1)
        # mask to padding
        mask_padded = target_hist_slice == counterpart_padding_value
        # final mask
        mask = mask_counterpart | mask_padded
        return mask

    def _hist_slicer(self, target_idx, target_hist, counterpart_padding_value):
        # target hist slice
        target_hist_slice = target_hist[target_idx]
        # calculate max hist in batch
        lengths = (target_hist_slice != counterpart_padding_value).sum(dim=1)
        max_len = lengths.max().item()
        # drop padding values
        target_hist_slice_trunc = target_hist_slice[:, :max_len]
        return target_hist_slice_trunc

    def _init_layers(self):
        self.embed_global = nn.Parameter(torch.randn(self.n_factors))

        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.embed_target = nn.Embedding(**kwargs)

        kwargs = dict(
            num_embeddings=self.n_items+1, 
            embedding_dim=self.n_factors,
            padding_idx=self.n_items,
        )
        self.embed_hist = nn.Embedding(**kwargs)

        nn.init.normal_(self.embed_global, mean=0.0, std=0.01)
        nn.init.normal_(self.embed_target.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.embed_hist.weight, mean=0.0, std=0.01)

        kwargs = dict(
            n_users=self.n_users,
            n_items=self.n_items,
            n_factors=self.n_factors*2,
            hidden=self.hidden,
            dropout=self.dropout,
        )
        self.affection = affection.Module(**kwargs)

        self.attn = AttentionMechanism()

        kwargs = dict(
            in_features=self.n_factors,
            out_features=1,
        )
        self.logit_layer = nn.Linear(**kwargs)

    def _assert_arg_error(self):
        CONDITION = (self.hidden[-1] == self.n_factors//2)
        ERROR_MESSAGE = f"Last MLP layer must match input size: {self.n_factors//2}"
        assert CONDITION, ERROR_MESSAGE