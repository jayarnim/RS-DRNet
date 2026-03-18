from .cat import Concatenation
from .prod import ElementwiseProduct


AGGREGATION_REGISTRY = {
    "cat": Concatenation,
    "prod": ElementwiseProduct,
}