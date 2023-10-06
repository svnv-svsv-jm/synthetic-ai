__all__ = [
    "data_tuple_to_kwargs",
    "get_soundstream",
    "get_semantictransformer",
    "get_coarsetransformer",
    "get_finetransformer",
    "configure_optimizer",
    "fetch_decode_waveform",
]

from loguru import logger
import typing as ty

import torchaudio
import torch
import torch.optim as optim
from torch import Tensor
from lightning.pytorch.utilities.types import OptimizerLRSchedulerConfig

from .audiolm_pytorch import (
    SoundStream,
    CoarseTransformer,
    AudioLMSoundStream,
    MusicLMSoundStream,
    SemanticTransformer,
    FineTransformer,
)
from .audiolm_pytorch.trainer import DATASET_FIELD_TYPE_CONFIG, determine_types


def fetch_decode_waveform(batch: ty.Dict[str, ty.Any], num_frames: int) -> torch.Tensor:
    """Fetch and decode the 1 - 2 seconds."""
    filename = batch["audio"]["path"][0]
    logger.trace(f"Fetching waveform from {filename}")
    waveform, _ = torchaudio.load(
        filename,
        frame_offset=0,
        num_frames=num_frames,
    )
    assert isinstance(waveform, torch.Tensor)
    return waveform


def configure_optimizer(model: torch.nn.Module, lr: float, patience: int) -> OptimizerLRSchedulerConfig:
    """Configure this model's optimizer and scheduler."""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        verbose=True,
        patience=patience,
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "monitor": "loss/train",
            "strict": False,
        },
    }


def data_tuple_to_kwargs(data: ty.Sequence[ty.Union[torch.Tensor, ty.List[str]]]) -> ty.Dict[str, ty.Any]:
    """Converts batch to dict."""
    ds_fields = determine_types(data, DATASET_FIELD_TYPE_CONFIG)
    x = dict(zip(ds_fields, data))
    logger.trace(f"{x.keys()}")
    return x


def get_finetransformer(
    finetransformer: ty.Union[FineTransformer, dict] = None,
    codebook_size: int = None,
) -> FineTransformer:
    """Initializes the CoarseTransformer model."""
    if isinstance(finetransformer, FineTransformer):
        model = finetransformer
    elif isinstance(finetransformer, dict):
        finetransformer["codebook_size"] = codebook_size
        model = FineTransformer(**finetransformer)
    else:
        assert codebook_size is not None
        model = FineTransformer(
            num_coarse_quantizers=3,
            num_fine_quantizers=9,
            codebook_size=codebook_size,
            dim=512,
            depth=6,
            flash_attn=True,
        )
    return model


def get_coarsetransformer(
    coarsetransformer: ty.Union[CoarseTransformer, dict] = None,
    codebook_size: int = None,
) -> CoarseTransformer:
    """Initializes the CoarseTransformer model."""
    if isinstance(coarsetransformer, CoarseTransformer):
        model = coarsetransformer
    elif isinstance(coarsetransformer, dict):
        coarsetransformer["codebook_size"] = codebook_size
        model = CoarseTransformer(**coarsetransformer)
    else:
        assert codebook_size is not None
        model = CoarseTransformer(
            num_semantic_tokens=codebook_size,
            codebook_size=codebook_size,  # 1024
            num_coarse_quantizers=3,
            dim=512,
            depth=6,
            flash_attn=True,
        )
    return model


def get_soundstream(
    soundstream: ty.Union[SoundStream, dict] = None,
    codebook_size: int = None,
    musiclm: bool = True,
) -> SoundStream:
    """Initializes the SoundStream model."""
    if isinstance(soundstream, SoundStream):
        model = soundstream
    elif isinstance(soundstream, dict):
        soundstream["codebook_size"] = codebook_size
        model = SoundStream(**soundstream)
    else:
        assert codebook_size is not None
        params = dict(
            codebook_size=codebook_size,  # 1024
            rq_num_quantizers=12,
            rq_groups=2,  # this paper proposes using multi-headed residual vector quantization - https://arxiv.org/abs/2305.02765
            attn_window_size=128,  # local attention receptive field at bottleneck
            attn_depth=2,  # 2 local attention transformer blocks - the soundstream folks were not experts with attention, so i took the liberty to add some. encodec went with lstms, but attention should be better
        )
        model = MusicLMSoundStream(**params) if musiclm else AudioLMSoundStream(**params)  # type: ignore
    return model


def get_semantictransformer(
    semantic_transformer: ty.Union[SemanticTransformer, dict] = None,
    codebook_size: int = None,
) -> SemanticTransformer:
    """Initializes the SemanticTransformer model."""
    if isinstance(semantic_transformer, SemanticTransformer):
        m = semantic_transformer
    elif isinstance(semantic_transformer, dict):
        m = SemanticTransformer(**semantic_transformer)
    else:
        assert codebook_size is not None
        m = SemanticTransformer(
            num_semantic_tokens=codebook_size,
            dim=1024,
            depth=6,
            flash_attn=True,
        )
    return m
