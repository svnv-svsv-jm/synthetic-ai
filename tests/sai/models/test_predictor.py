# pylint: disable=no-member,undefined-loop-variable
import pytest
from loguru import logger
import sys
import typing as ty

import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger

from sai.datasets import AisleDataset
from sai.models import CategoricalLSTMModel
from sai.utils import get_metric


def test_lstm(aisle_visits: pd.DataFrame) -> None:
    """Test it runs."""
    # Params
    input_is_one_hot = True
    # Data
    dataset = AisleDataset(aisle_visits, one_hot=input_is_one_hot)
    logger.info(f"dataset:\n{dataset}")
    # Loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in loader:
        assert isinstance(batch, Tensor)
        logger.info(f"Batch: {batch.size()}")
        logger.info(f"Values: {batch.min(), batch.max()}")
        break
    # Model
    if input_is_one_hot:
        vocab_size = batch.size(-1)  # Number of unique categories
    else:
        vocab_size = len(dataset.df["aisle"].unique())
    logger.info(f"vocab_size: {vocab_size}")
    model = CategoricalLSTMModel(vocab_size=vocab_size)
    logger.info(f"model:\n{model}")
    # Training
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=16,
        logger=[
            MLFlowLogger(  # type: ignore
                save_dir="pytest_artifacts",
                run_name="sequential",
                log_model=True,
            ),
            TensorBoardLogger(  # type: ignore
                save_dir="pytest_artifacts",
                name="sequential",
            ),
        ],
        enable_checkpointing=False,
    )
    # Tuner(trainer).lr_find(model, loader)  # type: ignore
    trainer.fit(model, loader)
    # Generate
    seq = model.generate_sequence(save_dir="pytest_artifacts/gen_seq")
    logger.info(f"Generated sequence: {seq}")
    # Utils
    metrics = get_metric(trainer, metric="loss/train")
    logger.info(metrics)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
