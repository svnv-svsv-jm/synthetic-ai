# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

import pandas as pd, numpy as np

from sai.datalab.datasets.loader import load_aisle, load_berka
from sai.datalab.benchmark import benchmark
from sai.datalab.synthesizers.flatautoencoder import FlatAutoEncoderSynthesizer


def test_benchmark() -> None:
    """Test it runs."""
    synthesizers = [FlatAutoEncoderSynthesizer]
    datasets = dict(berka=load_berka(), aisle=load_aisle())
    res = benchmark(synthesizers, datasets)
    logger.success(res)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
