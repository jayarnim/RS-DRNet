from ..config.model import (
    AffectionCfg,
    AssociationCfg,
    DRNetCfg,
)


def auto(cfg):
    model = cfg["model"]["name"]
    if model=="affection":
        return affection(cfg)
    elif model=="association":
        return association(cfg)
    elif model=="drnet":
        return drnet(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def affection(cfg):
    return AffectionCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        dropout=cfg["model"]["dropout"],
        agg=cfg["model"]["agg"],
    )


def association(cfg):
    return AssociationCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        beta=cfg["model"]["beta"],
        dropout=cfg["model"]["dropout"],
        agg=cfg["model"]["agg"],
    )


def drnet(cfg):
    affection = AffectionCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["affection"]["embedding_dim"],
        hidden_dim=cfg["model"]["affection"]["hidden_dim"],
        dropout=cfg["model"]["affection"]["dropout"],
        agg=cfg["model"]["affection"]["agg"],
    )
    association = AssociationCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        embedding_dim=cfg["model"]["association"]["embedding_dim"],
        hidden_dim=cfg["model"]["association"]["hidden_dim"],
        beta=cfg["model"]["association"]["beta"],
        dropout=cfg["model"]["association"]["dropout"],
        agg=cfg["model"]["association"]["agg"],
    )
    return DRNetCfg(
        affection=affection,
        association=association,
    )

