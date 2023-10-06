__all__ = ["MuLaNLightning"]

import typing as ty
from typing import Any
from loguru import logger

import torch
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
import lightning.pytorch.callbacks as cb

from .musiclm_pytorch import MuLaN as Mulan, AudioSpectrogramTransformer, TextTransformer, MuLaNEmbedQuantizer
from .utils import fetch_decode_waveform


class MuLaNLightning(pl.LightningModule):
    """MuLaN model."""

    def __init__(
        self,
        dim: int = 512,
        depth: int = 6,
        heads: int = 8,
        dim_head: int = 64,
        spec_n_fft: int = 128,
        spec_win_length: int = 24,
        spec_aug_stretch_factor: float = 0.8,
        lr: float = 1e-8,
        patience: int = 2,
        enable_checkpointing: bool = False,
        checkpoint_kwargs: ty.Dict[str, ty.Any] = {},
        num_frames: int = 1024,
    ) -> None:
        """
        Args:
            dim (int, optional): _description_. Defaults to 512.
            depth (int, optional): _description_. Defaults to 6.
            heads (int, optional): _description_. Defaults to 8.
            dim_head (int, optional): _description_. Defaults to 64.
            spec_n_fft (int, optional): _description_. Defaults to 128.
            spec_win_length (int, optional): _description_. Defaults to 24.
            spec_aug_stretch_factor (float, optional): _description_. Defaults to 0.8.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # Audio
        self.audio_transformer = AudioSpectrogramTransformer(  # type: ignore
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            spec_n_fft=spec_n_fft,
            spec_win_length=spec_win_length,
            spec_aug_stretch_factor=spec_aug_stretch_factor,
        )
        logger.trace(f"Initialized AudioSpectrogramTransformer: {self.audio_transformer}")
        # Text
        self.text_transformer = TextTransformer(  # type: ignore
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
        )
        logger.trace(f"Initialized TextTransformer: {self.text_transformer}")
        # MULAN
        self.model = Mulan(audio_transformer=self.audio_transformer, text_transformer=self.text_transformer)
        logger.trace(f"Initialized MuLaN: {self.model}")
        # Quantizer
        self.quantizer = MuLaNEmbedQuantizer(
            mulan=self.model,  # pass in trained mulan from above
            conditioning_dims=(dim, dim, dim),  # say all three transformers have model dimensions of 1024
            namespaces=("semantic", "coarse", "fine"),
        )

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

    def configure_callbacks(self) -> ty.List[pl.Callback]:  # type: ignore
        """Configure callbacks."""
        callbacks = []
        if self.hparams["enable_checkpointing"]:
            ckpt_kw: dict = self.hparams["checkpoint_kwargs"]
            ckpt_kw.setdefault("monitor", "loss/train")
            ckpt_kw.setdefault("mode", "min")
            ckpt_kw.setdefault("save_top_k", 3)
            ckpt_kw.setdefault("save_last", True)
            ckpt_kw.setdefault("save_on_train_epoch_end", True)
            logger.info(f"Creating {cb.ModelCheckpoint} with parameters: {ckpt_kw}")
            ckpt_cb_val: pl.Callback = cb.ModelCheckpoint(**ckpt_kw)
            callbacks.append(ckpt_cb_val)
        early = cb.EarlyStopping(
            monitor="loss/val",
            patience=5,
            mode="min",
            check_on_train_epoch_end=True,
            strict=False,
        )
        callbacks.append(early)
        return callbacks

    def forward(
        self,
        wavs: torch.Tensor = None,
        texts: ty.Sequence[str] = None,
        namespace: str = "semantic",
    ) -> ty.Tuple[ty.Optional[torch.Tensor], ty.Optional[torch.Tensor], ty.Optional[torch.Tensor]]:
        """Embeds input waveform and/or texts.
        Args:
            wavs (torch.Tensor, optional):
                Input waveform to embed. Defaults to None.
            texts (ty.Sequence[str], optional):
                Input texts to embed. Defaults to None.
            namespace (str, optional):
                Namespace for :class:`musiclm_pytorch.MuLaNEmbedQuantizer`. Defaults to "semantic".
        Returns:
            audio_embedding (torch.Tensor):
            text_embedding (torch.Tensor):
            conds (torch.Tensor):
        """
        audio_embedding: ty.Optional[torch.Tensor] = None
        text_embedding: ty.Optional[torch.Tensor] = None
        conds: ty.Optional[torch.Tensor] = None
        if wavs is not None:
            audio_embedding = self.model.get_audio_latents(wavs)  # type: ignore
            conds = self.quantizer(wavs=wavs, namespace=namespace)
        if texts is not None:
            text_embedding = self.model.get_text_latents(texts)
        return audio_embedding, text_embedding, conds

    def training_step(self, batch: ty.Dict[str, ty.Any], batch_idx: int) -> STEP_OUTPUT:
        """Training step."""
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: ty.Dict[str, ty.Any], batch_idx: int) -> STEP_OUTPUT:
        """Validation step."""
        return self.step(batch, batch_idx, "train")

    def test_step(self, batch: ty.Dict[str, ty.Any], batch_idx: int) -> STEP_OUTPUT:
        """Test step."""
        return self.step(batch, batch_idx, "train")

    def step(self, batch: ty.Dict[str, ty.Any], batch_idx: int, stage: str) -> STEP_OUTPUT:
        """Common step."""
        logger.trace(f"Batch ID: {batch_idx}")
        loss = self.loss(batch)
        self.log(f"loss/{stage}", loss, prog_bar=True)
        return loss

    def loss(self, batch: ty.Dict[str, ty.Any]) -> torch.Tensor:
        """Returns the loss over a batch."""
        logger.trace(f"Received: {batch}")
        waveform = fetch_decode_waveform(batch, self.hparams["num_frames"])
        loss: torch.Tensor = self.model(waveform, raw_texts=[batch["caption"][0]])
        return loss
