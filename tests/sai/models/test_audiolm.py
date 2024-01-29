# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

import lightning.pytorch as pl

from sai.models import AudioLMLightning


def test_audiolm() -> None:
    """Test we can initialize the model."""
    # Model
    model = AudioLMLightning(batch_size=4)
    # Test training
    pl.Trainer(fast_dev_run=True, accelerator="cpu").fit(model)
    # Generate
    generated_wav = model()
    logger.info(generated_wav)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
