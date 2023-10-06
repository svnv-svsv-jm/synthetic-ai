# pylint: disable=no-member
import pytest
from loguru import logger
import sys
import typing as ty

import pandas as pd, numpy as np

from sai.datalab.datasets.loader import load_aisle, load_berka
from sai.datalab.benchmark import benchmark, compare
from sai.datalab.synthesizers.flatautoencoder import FlatAutoEncoderSynthesizer


def test_training() -> None:
    """Test."""
    target_data = load_aisle()
    model = FlatAutoEncoderSynthesizer(number_of_epochs=1)
    model.fit(target_data=target_data)
    sample_size = 50
    sample_size = np.min([target_data.index.get_level_values("id").nunique(), sample_size])
    synthetic_data: pd.DataFrame = model.generate(int(sample_size))  # type: ignore
    categorical_features = [col for col in synthetic_data.columns if synthetic_data[col].dtype == "object"]
    for c in categorical_features:
        synthetic_data[c] = synthetic_data[c].astype("category")
    compare_vals = compare(target_data, synthetic_data)
    logger.success(compare_vals)


def test_benchmark() -> None:
    """Test it runs."""
    synthesizers = [FlatAutoEncoderSynthesizer]
    datasets = dict(berka=load_berka(), aisle=load_aisle())
    res = benchmark(synthesizers, datasets)  # type: ignore
    logger.success(res)


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
