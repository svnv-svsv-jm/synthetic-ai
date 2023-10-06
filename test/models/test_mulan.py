# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

from sai.datasets import MusicCaps
from sai.models import MuLaNLightning


def test_musiclm() -> None:
    """Test we can initialize the model."""
    # Load dataset
    dm = MusicCaps(
        samples_to_load=32,
        batch_size=1,
    )
    dm.prepare_data()
    dm.setup()
    logger.info(f"Sample: {dm.dataset[0]}")
    # Model
    model = MuLaNLightning()
    # Trainer
    trainer = pl.Trainer(
        logger=False,
        enable_checkpointing=False,
        max_steps=16,
        accelerator="cpu",
        overfit_batches=1,
    )
    # LR
    Tuner(trainer).lr_find(
        model,
        datamodule=dm,
        max_lr=1.0,
        min_lr=1e-12,
        update_attr=True,
    )
    # Train
    trainer.fit(model, dm)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
