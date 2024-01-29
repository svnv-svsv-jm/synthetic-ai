# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

import pandas as pd

from sai.datalab.datasets.loader import load_aisle
from sai.utils import get_dfs_to_have_common_columns, get_timeseries_frequencies


# @pytest.mark.parametrize("ckpt", ["checkpoints/TimeGAN/syn.csv", "checkpoints/flatautoencoder/syn.csv"])
# def test_quality_metrics(ckpt: str) -> None:
#     """Test it runs on synthetic data."""
#     # Load data
#     target_data = load_aisle()
#     synthetic_data = pd.read_csv(ckpt, index_col=["id", "sequence_pos"])
#     _testing_shapes(target_data, synthetic_data)
#     # Get frequencies
#     df_target = get_timeseries_frequencies(target_data)
#     df_syn = get_timeseries_frequencies(synthetic_data)
#     # Transform
#     df_syn = get_dfs_to_have_common_columns(df_syn, df_target, fill=0)
#     # Test shape
#     _testing_shapes(df_target, df_syn)


# def _testing_shapes(target_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> None:
#     """Helper to test for DF's shapes."""
#     c1 = target_data.reset_index().columns.sort_values()
#     c2 = synthetic_data.reset_index().columns.sort_values()
#     info = f"({len(c1)}) ({len(c2)}) | {c1} and {c2}"
#     logger.info(f"Will they match: {info}?")
#     assert len(c1) == len(c2), f"They do not match: {info}"
#     assert (c1 == c2).all(), f"They do not match:: {info}"


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
