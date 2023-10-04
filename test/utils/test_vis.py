# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

import pandas as pd

from sai.datalab.datasets.loader import load_aisle
from sai.utils import plot_token_frequencies, plot_token_positions


@pytest.mark.parametrize("ckpt", ["checkpoints/flatautoencoder/syn.csv"])
def test_vis(ckpt: str) -> None:
    """Test it runs on synthetic data."""
    # Load data
    target_data = load_aisle()
    synthetic_data = pd.read_csv(ckpt, index_col=["id", "sequence_pos"])
    # Test shape
    _testing_shapes(target_data, synthetic_data)
    # Get frequencies
    plot_token_frequencies(target_data, synthetic_data, show=False)
    # Get positions
    plot_token_positions(target_data, synthetic_data, "breakfast bakery", show=False)


def _testing_shapes(target_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
    """Helper to test for DF's shapes."""
    c1 = target_data.reset_index().columns.sort_values()
    c2 = synthetic_data.reset_index().columns.sort_values()
    info = f"({len(c1)}) ({len(c2)}) | {c1} and {c2}"
    logger.info(f"Will they match: {info}?")
    assert len(c1) == len(c2), f"They do not match: {info}"
    assert (c1 == c2).all(), f"They do not match:: {info}"


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
