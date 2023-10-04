__all__ = ["LLM"]

from loguru import logger
import typing as ty

from torch import Tensor
from lightning_transformers.task.nlp.language_modeling import LanguageModelingTransformer


class LLM(LanguageModelingTransformer):
    """LLM."""

    def __init__(self, *args: ty.Any, **kwargs: ty.Any) -> None:
        """."""
        super().__init__(*args, **kwargs)

    def _step(self, batch: ty.Any, batch_idx: int) -> Tensor:
        """Override."""
        logger.trace(f"({batch_idx}): {batch}")
        loss: Tensor = super()._step(batch, batch_idx)
        return loss.mean()
