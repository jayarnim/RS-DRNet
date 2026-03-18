from dataclasses import dataclass
from typing import Literal, Union
from .pipeline import PipelineCfg
from .trainer import TrainerCfg
from .evaluator import EvaluatorCfg
from .schema import SchemaCfg
from .model import AffectionCfg, AssociationCfg, DRNetCfg


@dataclass
class Config:
    model: Union[AffectionCfg, AssociationCfg, DRNetCfg]
    schema: SchemaCfg
    pipeline: PipelineCfg
    trainer: TrainerCfg
    evaluator: EvaluatorCfg
    strategy: Literal["pointwise", "pairwise", "listwise"]
    model_cls: Literal["affection", "association", "drnet"]
    dataset: str
    seed: int