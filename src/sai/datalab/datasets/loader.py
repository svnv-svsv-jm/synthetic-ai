__all__ = ["load_aisle", "load_cdnow", "load_berka", "load_mlb", "load_retail"]

from loguru import logger
from tqdm import tqdm
import pandas as pd
from os.path import dirname, join

from ..target_data_manipulation import prepare_common_data_format


def load_aisle(pad: bool = True) -> pd.DataFrame:
    """Load Aisle dataset. This is a dataset of aisle visits. It contains several sequences of aisle visits.
    Contains a table `aisle_visits`, that consists of 20,000 store visits, together with their 227,226 visited store aisles. In total there are 134 distinct aisles.
    """
    module_path = join(dirname(__file__), "data/")
    df = pd.read_csv(f"{module_path}aisle.csv")
    categorical_features = [col for col in df.columns if df[col].dtype == "object"]
    for c in categorical_features:
        df[c] = df[c].astype("category")
    df = prepare_common_data_format(df)
    # Pad
    if pad:
        max_sequence_length = int(len(df.reset_index()["sequence_pos"].unique()))
        logger.debug(f"Padding to {max_sequence_length}")
        padded_dfs = []
        for idx, group in tqdm(df.groupby("id")):
            padded_sequence = pd.DataFrame({"sequence_pos": list(range(max_sequence_length))})
            padded_sequence = padded_sequence.merge(group, on="sequence_pos", how="left")
            if idx == 0:
                logger.debug(padded_sequence.shape)
            padded_dfs.append(padded_sequence)
        df = pd.concat(padded_dfs, axis=0).reset_index().rename(columns={"index": "id"})
        df = df.drop(columns=["sequence_pos"])
        df = prepare_common_data_format(df).reset_index().rename(columns={"index": "id"})
        df[["tmp"]] = df[["sequence_pos"]]
        df[["sequence_pos"]] = df[["id"]]
        df[["id"]] = df[["tmp"]]
        df = df.drop(columns=["tmp"])
        logger.debug(df.shape)
        df = df.set_index(["id", "sequence_pos"]).sort_index()
    return df


def load_cdnow() -> pd.DataFrame:
    """
    load in CDNOW dataset
    :returns: pandas dataframe
    """
    module_path = join(dirname(__file__), "data/")
    return prepare_common_data_format(pd.read_csv(f"{module_path}cdnow_len5.csv"))


def load_berka() -> pd.DataFrame:
    """
    Load BERKA dataset
    :returns: pandas dataframe
    """
    module_path = join(dirname(__file__), "data/")
    df = pd.read_csv(f"{module_path}berka_len10.csv")
    categorical_features = [col for col in df.columns if df[col].dtype == "object"]
    for c in categorical_features:
        df[c] = df[c].astype("category")
    return prepare_common_data_format(df)


def load_mlb() -> pd.DataFrame:
    """
    Load MLB dataset
    :returns: pandas dataframe
    """
    module_path = join(dirname(__file__), "data/")
    return prepare_common_data_format(pd.read_csv(f"{module_path}mlb_len8.csv"))


def load_retail() -> pd.DataFrame:
    """
    Load RETAIL dataset
    :returns: pandas dataframe
    """
    module_path = join(dirname(__file__), "data/")
    return prepare_common_data_format(pd.read_csv(f"{module_path}retail_len100.csv.gz", compression="gzip"))
