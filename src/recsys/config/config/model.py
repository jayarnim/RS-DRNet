from dataclasses import dataclass


@dataclass
class AffectionCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class AssociationCfg:
    num_users: int
    num_items: int
    params: dict


@dataclass
class DRNetCfg:
    affection: AffectionCfg
    association: AssociationCfg