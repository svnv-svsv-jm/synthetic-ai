# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger, MLFlowLogger

from sai.datasets import AisleDataset
from sai.models import CategoricalLSTMModel
from sai.utils import get_metric


def test_lstm(aisle_visits: pd.DataFrame) -> None:
    """Test it runs."""
    # Params
    input_is_one_hot = True
    # Data
    dataset = AisleDataset(aisle_visits, one_hot=input_is_one_hot)
    logger.info(dataset)
    # Loader
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch in loader:
        assert isinstance(batch, Tensor)
        logger.info(f"Batch: {batch.size()}")
        logger.info(f"Values: {batch.min(), batch.max()}")
        break
    # Model
    vocab_size = len(dataset.df["aisle"].unique())  # Number of unique categories
    model = CategoricalLSTMModel(vocab_size=vocab_size)
    logger.info(model)
    # Training
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=16,
        logger=[
            MLFlowLogger(
                save_dir="pytest_artifacts",
                run_name="sequential",
                log_model=True,
            ),
            TensorBoardLogger(
                save_dir="pytest_artifacts",
                name="sequential",
            ),
        ],
        enable_checkpointing=False,
        auto_lr_find=True,
    )
    trainer.fit(model, loader)
    # Generate
    seq = model.generate_sequence(save_dir="pytest_artifacts/gen_seq")
    logger.info(f"Generated sequence: {seq}")
    # Utils
    metrics = get_metric(trainer, metric="loss/train")
    logger.info(metrics)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
