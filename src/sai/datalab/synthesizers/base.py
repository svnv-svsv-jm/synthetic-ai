# pylint: disable=unused-argument
import typing as ty
from loguru import logger
import pandas as pd
import torch

from ..synthesizers.utils import check_is_fitted, check_common_data_format

LOGGER = logger


class BaseSynthesizer(torch.nn.Module):
    """Base class for Synthesizer"""

    def __init__(self) -> None:
        super().__init__()

    def fit(self, target_data: pd.DataFrame, **kwargs: ty.Any) -> None:
        """
        Args:
            target_data (_type_): _description_
        """
        LOGGER.info(f"Training {self.__class__.__name__}")
        check_common_data_format(target_data)

    def generate(self, number_of_subjects: int, **kwargs: ty.Any) -> None:
        """Generate method.
        Args:
            number_of_subjects (int): _description_
        """
        LOGGER.info(f"Generating {self.__class__.__name__}")
        check_is_fitted(self)

    def forward(self, *args: ty.Any, **kwargs: ty.Any) -> ty.Any:
        """forward"""
        return self.generate(*args, **kwargs)
