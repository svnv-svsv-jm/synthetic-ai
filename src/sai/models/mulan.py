__all__ = ["MuLaN"]

import typing as ty
from typing import Any
from loguru import logger

import torch
import torch.optim as optim
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
import lightning.pytorch.callbacks as cb
from musiclm_pytorch import MuLaN as Mulan, AudioSpectrogramTransformer, TextTransformer, MuLaNEmbedQuantizer
import torchaudio


class MuLaN(pl.LightningModule):
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
        self.audio_transformer = AudioSpectrogramTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            spec_n_fft=spec_n_fft,
            spec_win_length=spec_win_length,
            spec_aug_stretch_factor=spec_aug_stretch_factor,
        )
        logger.trace(f"Initialized AudioSpectrogramTransformer: {self.audio_transformer}")
        self.text_transformer = TextTransformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head)
        logger.trace(f"Initialized TextTransformer: {self.text_transformer}")
        self.model = Mulan(audio_transformer=self.audio_transformer, text_transformer=self.text_transformer)
        logger.trace(f"Initialized MuLaN: {self.model}")
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
    ) -> ty.Tuple[torch.Tensor, ty.Optional[torch.Tensor]]:
        """Forward pass."""
        embeds: torch.Tensor
        conds: ty.Optional[torch.Tensor] = None
        if wavs is not None:
            embeds = self.model.get_audio_latents(wavs)
            conds = self.quantizer(wavs=wavs, namespace=namespace)
        else:
            embeds = self.model.get_text_latents(texts)
        return embeds, conds

    def training_step(self, batch: ty.Dict[str, ty.Any], batch_idx: int = None) -> STEP_OUTPUT:
        """Train a single training step."""
        logger.trace(f"Received: {batch}")
        waveform = self._fetch_waveform(batch)
        loss: torch.Tensor = self.model(waveform, raw_texts=[batch["caption"][0]])
        self.log("loss/train", loss, prog_bar=True)
        return loss

    def _fetch_waveform(self, batch: ty.Dict[str, ty.Any]) -> torch.Tensor:
        """Fetch and decode the 1 - 2 seconds."""
        frame_offset, num_frames = 0, 1024  # Fetch and decode the 1 - 2 seconds
        filename = batch["audio"]["path"][0]
        logger.trace(f"Fetching waveform from {filename}")
        waveform, _ = torchaudio.load(
            filename,
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        assert isinstance(waveform, torch.Tensor)
        return waveform
