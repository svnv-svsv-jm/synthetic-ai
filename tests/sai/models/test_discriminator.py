# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

from torch import Tensor
import lightning.pytorch as pl

from sai.datasets import DiscriminatorDataModule
from sai.models import Classifier


def test_discriminator() -> None:
    """Test it runs."""
    # Data
    dm = DiscriminatorDataModule(train_pct=0.8, batch_size=8)
    loader = dm.train_dataloader()
    logger.info(f"Loader: {len(loader)} - Dataset: {(len(dm.train))}")
    # Test data shapes and values are as expected
    for _, batch in enumerate(loader):
        X, y = batch
        assert isinstance(X, Tensor)
        assert isinstance(y, Tensor)
        logger.info(f"X: {X.size()}")
        logger.info(f"y: {y.size()}; min={y.min()}; max={y.max()}")
        assert 0 <= y.min() <= 1
        assert 0 <= y.max() <= 1
        break
    # Model
    in_features = int(X.size(-1))
    num_classes = 2
    n_layers = 10
    h_size = in_features * 10
    model = Classifier(
        in_features=in_features,
        num_classes=num_classes,
        hidden_dims=[h_size] * n_layers,  # this can be changed
        loss="bce",  # actually the nllloss is more stable but i want to be quick
    )
    # Training test
    trainer = pl.Trainer(
        accelerator="cpu",
        max_steps=512,
        logger=False,
    )
    trainer.fit(model, loader)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
