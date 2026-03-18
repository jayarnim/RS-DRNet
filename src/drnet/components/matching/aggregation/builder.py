from .aggregation.registry import AGGREGATION_REGISTRY


def aggregation_builder(name, **kwargs):
    cls = AGGREGATION_REGISTRY[name]
    return cls(**kwargs)