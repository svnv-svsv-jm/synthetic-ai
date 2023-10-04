# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

import pandas as pd
from torch import Tensor
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner

from sai.datasets import AisleDataModule
from sai.models import GAN


def test_timegan() -> None:
    """Test: we try to over-fit a small part of the training data and check the loss actually goes down, indicating nothing is stopping backpropagation."""
    # Params
    input_is_one_hot = True
    # Data
    dm = AisleDataModule(batch_size=32, one_hot=input_is_one_hot, lengths=[0.01, 0.99])
    # Loader
    loader = dm.train_dataloader()
    for batch in loader:
        assert isinstance(batch, Tensor)
        logger.info(f"Batch: {batch.size()}")
        logger.info(f"Values: {batch.min(), batch.max()}")
        break
    # Model
    vocab_size = len(dm.dataset.df["aisle"].unique())  # Number of unique categories
    model = GAN(vocab_size=vocab_size, input_is_one_hot=input_is_one_hot, lr=1e-4)
    logger.info(model)
    # Trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=2,
        logger=False,
        enable_checkpointing=False,
    )
    # Auto LR finder
    Tuner(trainer).lr_find(
        model,
        train_dataloaders=dm.train_dataloader(),
        val_dataloaders=dm.val_dataloader(),
        max_lr=10.0,
        min_lr=1e-6,
        update_attr=True,
    )
    # Test
    out = trainer.test(model, loader)
    logger.info(f"Test: {out}")
    loss_start = out[0]["loss-G/test"]
    trainer.fit(model, loader)
    out = trainer.test(model, loader)
    logger.info(f"Test: {out}")
    loss_end = out[0]["loss-G/test"]
    assert loss_end < loss_start, f"{loss_end} < {loss_start}"
    # Generate
    X = model.generate(pad_value=0, to_df=True)
    logger.info(X)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
