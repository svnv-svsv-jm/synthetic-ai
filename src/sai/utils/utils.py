__all__ = ["find_logger", "get_timeseries_frequencies", "get_dfs_to_have_common_columns"]

from loguru import logger
import typing as ty

from collections import Counter
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.loggers.logger import Logger
from pytorch_lightning.loggers.logger import Logger as Logger_
from sklearn.preprocessing import MinMaxScaler


def find_logger(
    this: ty.Union[pl.LightningModule, pl.Trainer],
    logger_type: ty.Type,
) -> ty.Optional[ty.Union[Logger, Logger_]]:
    """Checks if there is a logger of the type specified by the argument `logger`. For example:
    >>> isinstance(logger, (TensorBoardLogger, CSVLogger, MLFlowLogger))
    Args:
        this (pl.LightningModule | pl.Trainer):
            Model or Trainer.
        logger_type (Logger):
            A Pytorch Lightning logger class.
    """
    name = "pl_module" if isinstance(this, pl.LightningModule) else "trainer"
    # Try here
    if hasattr(this, "loggers") and isinstance(this.loggers, ty.Iterable):
        logger.debug(f"Looking for {logger_type} in {this.loggers}")
        for log in this.loggers:
            if isinstance(log, logger_type):
                assert isinstance(log, (Logger, Logger_))
                logger.debug(f"Returning {log}")
                return log
    logger.debug(f"No logger at {name}.loggers.")
    # And here
    if hasattr(this, "logger"):
        logger.debug(f"Looking for {logger_type} in {this.logger}")
        if isinstance(this.logger, logger_type):
            assert isinstance(this.logger, (Logger, Logger_))
            logger.debug(f"Returning {this.logger}")
            return this.logger
    logger.debug(f"No logger at {name}.logger.")
    return None


def get_timeseries_frequencies(
    data: pd.DataFrame,
    col_name: str = "aisle",
    index_name: str = "sequence_pos",
    normalize: bool = True,
) -> pd.DataFrame:
    """."""
    logger.trace(data.head())
    # For each token, get an histogram of appereance by position in the sequence
    hist = dict()
    for idx, group in data.groupby(col_name):
        x = list(group.reset_index()[index_name])
        x.sort()
        hist[idx] = x
    data = dict()
    for key, list_of_app in hist.items():
        counter = Counter(list_of_app)
        data[key] = counter
    df = pd.DataFrame(data).fillna(0)
    if normalize:
        scaler = MinMaxScaler()
        scaler.fit(df)
        # Transform the DataFrame to normalize the values
        normalized_data = scaler.transform(df)
        # Create a new DataFrame with normalized values
        df = pd.DataFrame(normalized_data, columns=df.columns)
    return df


def get_dfs_to_have_common_columns(df1: pd.DataFrame, df2: pd.DataFrame, fill: ty.Any = None) -> pd.DataFrame:
    """Extends one dataframe `df1` to have the columns of the other `df2`, and fills it with value `fill`."""
    missing_columns = list(set(df2.columns) - set(df1.columns))
    logger.debug(f"Missing columns: {len(missing_columns)}")
    for c in missing_columns:
        logger.trace(f"Filling {c}")
        df1[c] = fill
    return df1
