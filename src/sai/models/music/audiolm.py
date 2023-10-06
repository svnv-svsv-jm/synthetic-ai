__all__ = ["AudioLMLightning"]

import typing as ty
from typing import Any
from loguru import logger

from pathlib import Path
import lightning.pytorch as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT, OptimizerLRSchedulerConfig
import torch
import torch.optim as optim
from torch import Tensor
from torch.utils.data import random_split, Subset, DataLoader, Dataset

from sai.datasets import load_musiccaps
from .audiolm_pytorch.data import SoundDataset, get_dataloader
from .audiolm_pytorch import (
    AudioLM as AudioLM_,
    HubertWithKmeans,
    SoundStream,
    SemanticTransformer,
    SemanticTransformerWrapper,
    CoarseTransformer,
    CoarseTransformerWrapper,
    FineTransformer,
    FineTransformerWrapper,
)
from .hubert import download_hubert
from .loss import loss_soundstream, loss_transformer
from .utils import (
    fetch_decode_waveform,
    get_soundstream,
    get_semantictransformer,
    get_coarsetransformer,
    get_finetransformer,
    configure_optimizer,
)


class AudioLMLightning(pl.LightningModule):
    """AudioLM model."""

    def __init__(
        self,
        soundstream: ty.Union[SoundStream, dict] = None,
        semantic_transformer: ty.Union[SemanticTransformer, dict] = None,
        coarse_transformer: ty.Union[CoarseTransformer, dict] = None,
        fine_transformer: ty.Union[FineTransformer, dict] = None,
        audio_conditioner: torch.nn.Module = None,
        num_frames: int = 1024,
        lr: float = 1e-8,
        patience: int = 2,
        data_folder: str = ".data/music_data",
        split: ty.Sequence[ty.Union[int, float]] = (0.5, 0.3, 0.2),
        batch_size: int = 1,
        data_max_length_seconds: int = 2,
    ) -> None:
        """_summary_

        Args:
            soundstream (SoundStream):
                SoundStream model.
            semantic_transformer (ty.Union[SemanticTransformer, dict], optional): _description_. Defaults to None.
            coarse_transformer (ty.Union[CoarseTransformer, dict], optional): _description_. Defaults to None.
            fine_transformer (ty.Union[FineTransformer, dict], optional): _description_. Defaults to None.
            audio_conditioner (torch.nn.Module, optional): _description_. Defaults to None.
            num_frames (int, optional): _description_. Defaults to 1024.
            lr (float, optional): _description_. Defaults to 1e-8.
            patience (int, optional): _description_. Defaults to 2.
            data_folder (str, optional):
                Folder containing audio files. Defaults to ".data/music_data".
            split (ty.Sequence[ty.Union[int, float]], optional): _description_. Defaults to (0.5, 0.3, 0.2).
            batch_size (int, optional): _description_. Defaults to 1.
            data_max_length_seconds (int, optional): _description_. Defaults to 2.
        """
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # Wave2vec
        self.wave2vec: HubertWithKmeans = download_hubert()
        self.codebook_size: int = self.wave2vec.codebook_size  # type: ignore
        # SoundStream
        self.soundstream: SoundStream = get_soundstream(soundstream, self.codebook_size)
        # Transformers: need to be wrapped, for some weird reason. This was decided by the original authors of `audiolm_pytorch`
        # Semantic Transformer
        self.semantic_transformer = get_semantictransformer(semantic_transformer, self.codebook_size)
        self.semantic_transformer_wrapper = SemanticTransformerWrapper(
            wav2vec=self.wave2vec,
            transformer=self.semantic_transformer,
            audio_conditioner=audio_conditioner,
        )
        # Coarse Transformer
        self.coarse_transformer = get_coarsetransformer(coarse_transformer, self.codebook_size)
        self.coarse_transformer_wrapper = CoarseTransformerWrapper(
            wav2vec=self.wave2vec,
            codec=self.soundstream,
            transformer=self.coarse_transformer,
            audio_conditioner=audio_conditioner,
        )
        # Fine Transformer
        self.fine_transformer = get_finetransformer(fine_transformer, self.codebook_size)
        self.fine_transformer_wrapper = FineTransformerWrapper(
            codec=self.soundstream,
            transformer=self.fine_transformer,
            audio_conditioner=audio_conditioner,
        )
        # AudioLM
        self.model = AudioLM_(
            wav2vec=self.wave2vec,
            codec=self.soundstream,
            semantic_transformer=self.semantic_transformer,
            coarse_transformer=self.coarse_transformer,
            fine_transformer=self.fine_transformer,
        )
        # Losses
        self.loss_st: torch.Tensor
        self.loss_stf: torch.Tensor
        self.loss_ctf: torch.Tensor
        self.loss_ftf: torch.Tensor
        # Data
        self.dataset: SoundDataset
        self.ds_train: Subset
        self.ds_val: Subset
        self.ds_test: Subset
        self.split = split

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def prepare_data(self) -> None:
        """Prepare dataset."""
        # Check folder
        data_folder = self.hparams["data_folder"]
        if not Path(data_folder).exists():
            logger.debug("Downloading data...")
            load_musiccaps(root=data_folder)
        # Create dataset
        data_max_length_seconds = self.hparams["data_max_length_seconds"]
        data_max_length = int(data_max_length_seconds * self.soundstream.target_sample_hz)
        self.dataset = SoundDataset(
            data_folder,
            max_length=data_max_length,
            target_sample_hz=self.soundstream.target_sample_hz,
            seq_len_multiple_of=self.soundstream.seq_len_multiple_of,
        )

    def setup(self, stage: str = None) -> None:
        """Creates training, val, test splits."""
        if not hasattr(self, "dataset"):
            self.prepare_data()
        self.ds_train, self.ds_val, self.ds_test = random_split(
            self.dataset,
            lengths=self.split,
        )
        logger.debug(f"Created training set of size: {len(self.ds_train)}")
        logger.debug(f"Created validation set of size: {len(self.ds_val)}")
        logger.debug(f"Created test set of size: {len(self.ds_test)}")

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.ds_train)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.ds_val)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.ds_test)

    def _loader(self, ds: Dataset) -> DataLoader:
        """Create DataLoader from Dataset."""
        batch_size = self.hparams["batch_size"]
        return get_dataloader(ds, batch_size=batch_size, drop_last=True, shuffle=True)

    def configure_optimizers(self) -> OptimizerLRSchedulerConfig:  # type: ignore
        """Configure this model's optimizer and scheduler."""
        lr = self.lr
        patience = self.hparams["patience"]
        # config = [
        #     configure_optimizer(self.soundstream, lr, patience),
        #     configure_optimizer(self.semantic_transformer, lr, patience),
        #     configure_optimizer(self.coarse_transformer, lr, patience),
        #     configure_optimizer(self.fine_transformer, lr, patience),
        # ]
        return configure_optimizer(self, lr, patience)

    def forward(
        self,
        batch_size: int = 1,
        prime_wave: Tensor = None,
        text: ty.Sequence[str] = None,
    ) -> Tensor:
        """Generate Audio."""
        generated_wav: Tensor
        # Priming
        if prime_wave is not None:
            if prime_wave.dim() == 1:
                prime_wave = prime_wave.unsqueeze(0)
            assert prime_wave.dim() == 2, f"Wrong dim: {prime_wave.size()}. Shape must be (B,L)."
            generated_wav = self.model(prime_wave=prime_wave)
            return generated_wav
        # Text condition, if given
        if text is not None:
            if isinstance(text, str):
                text = [text]
            generated_wav = self.model(text=text)
            return generated_wav
        # No conditioning
        generated_wav = self.model(batch_size=batch_size)
        return generated_wav

    def training_step(
        self,
        batch: ty.Union[Tensor, ty.Sequence[Tensor]],
        batch_idx: int,
        optimizer_idx: int = None,
    ) -> STEP_OUTPUT:
        """Training step."""
        return self.step(batch, batch_idx, optimizer_idx, "train")

    def validation_step(
        self,
        batch: ty.Union[Tensor, ty.Sequence[Tensor]],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """Validation step."""
        return self.step(batch, batch_idx, None, "val")

    def test_step(
        self,
        batch: ty.Union[Tensor, ty.Sequence[Tensor]],
        batch_idx: int,
    ) -> STEP_OUTPUT:
        """Test step."""
        return self.step(batch, batch_idx, None, "test")

    def step(
        self,
        batch: ty.Union[Tensor, ty.Sequence[Tensor]],
        batch_idx: int = None,
        optimizer_idx: int = None,
        stage: str = "train",
    ) -> STEP_OUTPUT:
        """Step."""
        logger.trace(f"[{batch_idx}] batch={type(batch)} optimizer_idx={optimizer_idx} stage={stage}")
        if isinstance(batch, Tensor):
            batch = [batch]
        loss = torch.Tensor([0.0]).to(self.device)
        # waveform = batch  # fetch_decode_waveform(batch, self.hparams["num_frames"])
        for waveform in batch:
            logger.trace(f"Wave: {waveform.size()}")
            assert waveform.size(-1) > 0, f"??? {waveform.size()}"
            # SoundStream
            if optimizer_idx == 0 or optimizer_idx is None:
                (
                    self.loss_st,
                    recon_loss,
                    multi_spectral_recon_loss,
                    adversarial_loss,
                    feature_loss,
                    all_commitment_loss,
                ) = loss_soundstream(self.soundstream, waveform)
                loss += self.loss_st
                self.log(f"loss_st/{stage}", self.loss_st)
                self.log(f"recon_loss/{stage}", recon_loss)
                self.log(f"multi_spectral_recon_loss/{stage}", multi_spectral_recon_loss)
                self.log(f"adversarial_loss/{stage}", adversarial_loss)
                self.log(f"feature_loss/{stage}", feature_loss)
                self.log(f"all_commitment_loss/{stage}", all_commitment_loss)
            # Transformers
            if optimizer_idx == 1 or optimizer_idx is None:
                self.loss_stf = loss_transformer(self.semantic_transformer_wrapper, (waveform,))
                loss += self.loss_stf
                self.log(f"loss_stf/{stage}", self.loss_stf)
            if optimizer_idx == 2 or optimizer_idx is None:
                self.loss_ctf = loss_transformer(self.coarse_transformer_wrapper, (waveform,))
                loss += self.loss_ctf
                self.log(f"loss_ctf/{stage}", self.loss_ctf)
            if optimizer_idx == 3 or optimizer_idx is None:
                self.loss_ftf = loss_transformer(self.fine_transformer_wrapper, (waveform,))
                loss += self.loss_ftf
                self.log(f"loss_ftf/{stage}", self.loss_ftf)
        # Return
        return loss
