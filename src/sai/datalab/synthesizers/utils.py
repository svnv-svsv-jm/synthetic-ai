from typing import Any
from pandas import DataFrame
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
import pandas as pd

from ..utils.exceptions import NotFittedError


def check_is_fitted(generator: Any) -> None:
    """
    Perform validation that generator is fitted

    Inspired by sklearn.

    # https://github.com/scikit-learn/scikit-learn/blob/c9f8f52a781ddd477bd6a1e5dce19a88d1176349/sklearn/utils/validation.py#L958

    This is meant to be used internally by generators in their generate function.

    A generator is considered to be fitted if there exists attributes the ends with underscores
    and does not start with double underscore.

    :param self: generator instance

    :returns: None

    :raises: RuntimeErroNotFittedErrorr

    """

    attrs = [v for v in vars(generator) if v.endswith("_") and not v.startswith("__")]

    if not attrs:
        raise NotFittedError("{} is not fitted".format(type(generator).__name__))


def check_common_data_format(data: pd.DataFrame) -> None:
    """
    Perform validation for common data format. Used to screen input to validate if data is in common data format

    Requirements:
    * Pandas DataFrame
    * numerical and cat dtypes
    * must have id and column as multi-index


    :param data: Pandas Dataframe

    :raise: AssertionError if any criteria is not met

    """

    assert isinstance(data, DataFrame), "Data is not Pandas DataFrame"

    column_types = dict(zip(data.dtypes.index, data.dtypes))

    column_type_dict = {}
    true_vector = []

    for col_name, dtype in column_types.items():
        if is_numeric_dtype(dtype):
            column_type_dict[col_name] = "numeric"
            true_vector.append(True)
        elif is_categorical_dtype(dtype):
            column_type_dict[col_name] = "categorical"
            true_vector.append(True)
        else:
            true_vector.append(False)

    assert sum(true_vector) == len(
        data.columns
    ), f"Data must only have numeric and categorical column types but we have {column_types}"

    index_names = data.index.names
    assert "id" in index_names, f"`id` not in Index: {index_names}"


def _assign_column_type(data: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """_summary_
    Args:
        data (pd.DataFrame): _description_
        reference (pd.DataFrame): _description_
    Returns:
        pd.DataFrame: _description_
    """
    data_copy = data.copy()
    for col in data_copy:
        col_type = reference[col]
        if col_type == "category":
            f = pd.Categorical
        elif col_type == "number":
            f = pd.to_numeric
        data_copy.loc[:, col] = f(data_copy.loc[:, col])
    return data_copy
