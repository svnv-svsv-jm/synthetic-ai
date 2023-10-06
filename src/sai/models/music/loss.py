__all__ = ["loss_soundstream", "loss_transformer"]

from loguru import logger
import typing as ty

import torch
from torch import Tensor
from .audiolm_pytorch import (
    SoundStream,
    CoarseTransformerWrapper,
    SemanticTransformerWrapper,
    FineTransformerWrapper,
)

from .utils import data_tuple_to_kwargs


def loss_soundstream(
    soundstream: SoundStream,
    wave: Tensor,
) -> ty.Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Soundstream loss."""
    loss, (
        recon_loss,
        multi_spectral_recon_loss,
        adversarial_loss,
        feature_loss,
        all_commitment_loss,
    ) = soundstream(wave, return_loss_breakdown=True)
    return (
        loss,
        recon_loss,
        multi_spectral_recon_loss,
        adversarial_loss,
        feature_loss,
        all_commitment_loss,
    )


def loss_transformer(
    transformer: ty.Union[SemanticTransformerWrapper, CoarseTransformerWrapper, FineTransformerWrapper],
    data: ty.Sequence[ty.Union[Tensor, ty.List[str]]],
) -> Tensor:
    """Loss for all Transformers."""
    data_kwargs = data_tuple_to_kwargs(data)
    logger.trace(f"Got data: {data_kwargs.keys()}")
    loss: Tensor = transformer(**data_kwargs, return_loss=True)
    return loss
