from ..config.model import AffectionCfg, AssociationCfg, DRNetCfg


def model(cfg):
    cls = cfg["model"]["name"]

    if cls=="affection":
        return affection(cfg)
    elif cls=="association":
        return association(cfg)
    elif cls=="drnet":
        return drnet(cfg)
    else:
        raise ValueError("invalid model name in .yaml config")


def affection(cfg):
    return AffectionCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def association(cfg):
    return AssociationCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"],
    )


def drnet(cfg):
    affection = AffectionCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["affection"],
    )
    association = AssociationCfg(
        num_users=cfg["data"]["entity"]["num_users"],
        num_items=cfg["data"]["entity"]["num_items"],
        params=cfg["model"]["params"]["association"],
    )
    return DRNetCfg(
        affection=affection,
        association=association,
    )