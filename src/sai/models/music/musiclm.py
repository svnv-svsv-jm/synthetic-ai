__all__ = ["MusicLM"]

import typing as ty
from loguru import logger

import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
import torch
import torch.optim as optim

from .musiclm_pytorch import MusicLM as MusicLM_, MuLaNEmbedQuantizer
from .audiolm_pytorch import AudioLM
from .mulan import MuLaNLightning


class MusicLM(pl.LightningModule):
    """MusicLM model."""

    def __init__(
        self,
        audio_lm: AudioLM,
        mulan: MuLaNLightning,
    ) -> None:
        """
        Args:
            audio_lm (AudioLM):
                `AudioLM` from https://github.com/lucidrains/audiolm-pytorch. Usually initialized as:
                >>> audiolm = AudioLM(
                >>>     wav2vec = wav2vec,
                >>>     codec = soundstream,
                >>>     semantic_transformer = semantic_transformer,
                >>>     coarse_transformer = coarse_transformer,
                >>>     fine_transformer = fine_transformer
                >>> )
            mulan (MuLaN):
                The `MuLaN` model with a :class:`MuLaNEmbedQuantizer`.
        """
        super().__init__()
        # MuLaN
        self.mulan = mulan
        assert isinstance(mulan.quantizer, MuLaNEmbedQuantizer)
        # Backbone MusicLM
        self.model = MusicLM_(audio_lm=audio_lm, mulan_embed_quantizer=mulan.quantizer)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:
        """Configure this model's optimizer and scheduler."""
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="min",
            verbose=True,
            patience=self.hparams["patience"],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train",
                "strict": False,
            },
        }

    def forward(self, text: str, num_samples: int = 1, **audio_lm_kwargs: ty.Any) -> torch.Tensor:
        """Generate music from a prompt."""
        music: torch.Tensor = self.model(text, num_samples=num_samples, **audio_lm_kwargs)
        return music

    def training_step(self, batch: ty.Dict[str, ty.Any], batch_idx: int) -> STEP_OUTPUT:
        """Training step."""
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: ty.Dict[str, ty.Any], batch_idx: int) -> STEP_OUTPUT:
        """Validation step."""
        return self.step(batch, batch_idx, "val")

    def test_step(self, batch: ty.Dict[str, ty.Any], batch_idx: int) -> STEP_OUTPUT:
        """Test step."""
        return self.step(batch, batch_idx, "test")

    def step(self, batch: ty.Dict[str, ty.Any], batch_idx: int, stage: str) -> STEP_OUTPUT:
        """Common step."""
        logger.trace(f"Batch ID: {batch_idx}")
        loss = self.loss(batch)
        self.log(f"loss/{stage}", loss, prog_bar=True)
        return loss

    def loss(self, batch: ty.Dict[str, ty.Any]) -> torch.Tensor:
        """Returns the loss over a batch."""
        logger.trace(f"Received: {batch}")
        loss_mulan = self.mulan.loss(batch)
        return loss_mulan
