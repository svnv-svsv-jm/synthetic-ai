# pylint: disable=no-member,undefined-loop-variable
import pytest
from loguru import logger
import sys
import typing as ty

from torch import Tensor
import lightning.pytorch as pl
import pandas as pd

from sai.datasets import AisleDataModule
from sai.models import TimeGAN


def test_timegan() -> None:
    """Test: we try to over-fit a small part of the training data and check the loss actually goes down, indicating nothing is stopping backpropagation."""
    # Params
    input_is_one_hot = True
    # Data
    dm = AisleDataModule(
        batch_size=64,
        one_hot=input_is_one_hot,
        lengths=[0.2, 0.8],
    )
    # Loader
    loader = dm.train_dataloader()
    for batch in loader:
        assert isinstance(batch, Tensor)
        logger.info(f"Batch: {batch.size()}")
        logger.info(f"Values: {batch.min(), batch.max()}")
        break
    # Model
    vocab_size = batch.size(-1)
    model = TimeGAN(
        vocab_size=vocab_size,
        input_is_one_hot=input_is_one_hot,
        lr=1e-2,
    )
    logger.info(f"model:\n{model}")
    # Generate
    X = model.generate(pad_value=0, sequence_length=32, to_df=True)
    assert isinstance(X, pd.DataFrame)
    logger.info(f"Generated data:\n{X.head()}")
    # Trainer
    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=3,
        logger=False,
        enable_checkpointing=False,
    )
    # Test
    out = trainer.test(model, loader)
    logger.info(f"Test: {out}")
    loss_start = out[0]["loss-G/test"]
    trainer.fit(model, loader)
    out = trainer.test(model, loader)
    logger.info(f"Test: {out}")
    loss_end = out[0]["loss-G/test"]
    logger.info(f"{loss_end} < {loss_start}")
    assert loss_end < loss_start, f"{loss_end} < {loss_start}"


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
