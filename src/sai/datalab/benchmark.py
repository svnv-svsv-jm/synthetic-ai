"""

Compare + Benchmark

Compare - calculate benchmark function for one type of metric
benchmark - compare multiple synthesizers over multiple datasets
"""

__all__ = ["benchmark"]

from loguru import logger
from typing import Type, List, Callable, Dict, Union, Sequence
from pandas import DataFrame
import numpy as np
import pandas as pd

import time

from .synthesizers.base import BaseSynthesizer
from .synthesizers.flatautoencoder import FlatAutoEncoderSynthesizer
from .synthesizers.identity import IdentitySynthesizer
from .synthesizers.shuffle import ShuffleSynthesizer
from .metrics import compare
from .datasets.loader import load_cdnow, load_berka, load_mlb


SYN_CLASSES = Union[
    Type[BaseSynthesizer],
    Type[FlatAutoEncoderSynthesizer],
    Type[IdentitySynthesizer],
    Type[ShuffleSynthesizer],
]


def benchmark(
    synthetizers_classes: Sequence[SYN_CLASSES],
    datasets: Dict[str, DataFrame] = {},
    verbose: bool = False,
    log: bool = True,
    sample_size: int = 50,
) -> pd.DataFrame:
    """
    One function to rule them all.

    Train , generate and analyze specific datasets for a given set of synthesizer classes.

    :param synthetizers_classes: list of instance of synthesizer classes
    :param datasets: dict keys: dataset name dict values : pandas dataframe in common data format
    :param verbose: verbose benchmarking

    :returns: table of results
    """
    if len(datasets) == 0:
        # put all
        cdnow = load_cdnow()
        berka = load_berka()
        mlb = load_mlb()
        datasets = {"cdnow": cdnow, "berka": berka, "mlb": mlb}

    results_list = []
    for synthetizers_class in synthetizers_classes:
        syn_name = synthetizers_class.__name__
        logger.info(f"Evaluating synthesizer {syn_name}")

        for dataset_name, dataset in datasets.items():
            if verbose or log:
                logger.info(f"With dataset {dataset_name}")
            target_data = dataset
            start = time.time()
            # memory consumption
            #  ｡ﾟ☆: *.☽ .* :☆ﾟ
            logger.info(f"Training {synthetizers_class}...")
            model = synthetizers_class()
            model.fit(target_data=target_data)
            target_data_class = target_data
            #  ｡ﾟ☆: *.☽ .* :☆ﾟ
            end = time.time()
            diff = np.round(end - start)
            if verbose or log:
                logger.info(f"Training took {diff} seconds")
            start = time.time()
            #  ｡ﾟ☆: *.☽ .* :☆ﾟ
            # synthetic_data = syntheizers_class.generate(len(target_data))
            logger.info(f"Generating with {synthetizers_class}...")
            sample_size = np.min([target_data.index.get_level_values("id").nunique(), sample_size])
            logger.info(f"Generation of {sample_size}")
            synthetic_data: pd.DataFrame = model.generate(int(sample_size))  # type: ignore
            categorical_features = [
                col for col in synthetic_data.columns if synthetic_data[col].dtype == "object"
            ]
            for c in categorical_features:
                synthetic_data[c] = synthetic_data[c].astype("category")
            #  ｡ﾟ☆: *.☽ .* :☆ﾟ
            end = time.time()
            diff = np.round(end - start)
            if verbose or log:
                logger.info(f"Generating took {diff} seconds")
            start = time.time()
            #  ｡ﾟ☆: *.☽ .* :☆ﾟ
            # need to compare exactly what the generator sees
            logger.info("Comparing...")
            compare_vals = compare(target_data_class, synthetic_data)
            df_results = pd.DataFrame([syn_name, dataset_name] + list(compare_vals.values())).T
            df_results.columns = ["synthesizer", "dataset_name"] + list(compare_vals.keys())
            results_list.append(df_results)
            #  ｡ﾟ☆: *.☽ .* :☆ﾟ
            end = time.time()
            diff = np.round(end - start)
            logger.info(f"Comparing took {diff} seconds")

    if len(results_list) > 1:
        df_results_all = pd.concat(results_list)
    elif len(results_list) == 1:
        df_results_all = df_results
    else:
        raise RuntimeError(f"Nothing happened... {results_list}")

    df_results_all = df_results_all.sort_values("dataset_name").set_index("dataset_name")

    return df_results_all
