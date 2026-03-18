from dataclasses import dataclass


@dataclass
class AffectionCfg:
    num_users: int
    num_items: int
    embedding_dim: int
    hidden_dim: list
    dropout: float
    agg: str


@dataclass
class AssociationCfg:
    num_users: int
    num_items: int
    embedding_dim: int
    hidden_dim: list
    beta: float
    dropout: float
    agg: str


@dataclass
class DRNetCfg:
    affection: AffectionCfg
    association: AssociationCfg