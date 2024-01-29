__all__ = ["DiscriminatorDataset", "DiscriminatorDataModule", "AisleDataset", "AisleDataModule", "df_to_cat"]

from loguru import logger
import typing as ty
import pandas as pd
import numpy as np
from category_encoders import OrdinalEncoder
import sqlite3
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

from sai.datalab.datasets.loader import load_aisle
from sai.utils import f_one_hot

DATASET_TYPE = ty.Union[pd.DataFrame, np.ndarray, Dataset]


def to_torch(
    X: pd.DataFrame,
    categorical_features: ty.List[str] = None,
) -> Tensor:
    """Converts a DataFrame to Torch array."""
    if categorical_features is not None:
        numerical_features = [x for x in X.columns if x in categorical_features]
        categorical_features = [x for x in categorical_features if x not in ["label"]]
        labels = X["label"]
        X_cat = X[categorical_features]
        X_num = X[numerical_features]
        X_cat = torch.tensor(np.asarray(X_cat)).float()
        X_num = torch.tensor(np.asarray(X_num)).float()
        labels = torch.tensor(np.asarray(labels)).float()
        X_cat = f_one_hot(X_cat)
        X = torch.cat((X_cat, X_num, labels), dim=-1)
        assert isinstance(X, Tensor)
        return X
    X = np.asarray(X).astype(np.float32)
    X = torch.tensor(X)
    assert isinstance(X, Tensor)
    return X


class DiscriminatorDataset(Dataset):
    """Dataset class for our discriminator."""

    def __init__(
        self,
        X: pd.DataFrame,
        categorical_features: ty.List[str] = None,
    ) -> None:
        """
        Args:
            X (pd.DataFrame): _description_
            categorical_features (ty.List[str], optional): _description_. Defaults to None.
        """
        super().__init__()
        self.X = to_torch(X, categorical_features)

    def __len__(self) -> int:
        """Length of the dataset: sum of the two."""
        return int(self.X.size(0))

    def __getitem__(self, idx: int) -> ty.Tuple[Tensor, Tensor]:
        """Get one sample."""
        x = self.X[idx, :-1].view(-1)
        y = self.X[idx, -1].view(-1)
        return x, y.long()


class DiscriminatorDataModule(pl.LightningDataModule):
    """Dataset class for our discriminator."""

    def __init__(
        self,
        train_pct: float = 0.8,
        train_data_path: str = "data/census-original.csv",
        val_data_path: str = "data/census-synthetic.csv",
        batch_size: int = 8,
        num_workers: int = 1,
    ) -> None:
        super().__init__()
        self.num_workers = num_workers
        self.batch_size = batch_size
        # Init
        logger.debug("Loading files...")
        X_true = pd.read_csv(train_data_path)
        X_syn = pd.read_csv(val_data_path)
        # Handle categorical
        logger.debug("Handling categorical values...")
        categorical_features = [col for col in X_true.columns if X_true[col].dtype == "object"]
        encoder = OrdinalEncoder(
            cols=categorical_features,
            handle_unknown="ignore",
            return_df=True,
        ).fit(X_true)
        X_true_df = encoder.transform(X_true)
        X_syn_df = encoder.transform(X_syn)
        X_true_df["label"] = True
        X_syn_df["label"] = False
        # Concat
        logger.debug("Creating dataset...")
        self.df = pd.concat([X_true_df, X_syn_df])
        self.dataset = DiscriminatorDataset(self.df)
        # Split
        logger.debug("Random split...")
        n = len(self.dataset)  # type: ignore
        n_train = int(train_pct * n)
        n_val = n - n_train
        self.train, self.test = random_split(self.dataset, [n_train, n_val])  # type: ignore

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


class AisleDataset(Dataset):
    """Aisle dataset."""

    def __init__(
        self,
        data: pd.DataFrame = None,
        timeseries_identifier: str = "id",
        one_hot: bool = False,
    ):
        if data is None:
            df = load_aisle(pad=False).reset_index()
        else:
            assert isinstance(data, pd.DataFrame), f"Please provide a {pd.DataFrame}, and not {type(data)}."
            df = data
        df, categorical_features = df_to_cat(df)
        self.encoder = OrdinalEncoder(
            cols=categorical_features,
            handle_unknown="ignore",
            return_df=True,
        ).fit(df)
        df = self.encoder.transform(df)
        self.df, _ = df_to_cat(df)
        assert (
            timeseries_identifier in self.df.columns
        ), f"{timeseries_identifier} not found in {self.df.columns}. Check out if it is in the index: {self.df.reset_index().columns}?"
        self.longest_seq = self.df[timeseries_identifier].value_counts().max()
        tensors = []
        for _, group in enumerate(self.df.groupby(timeseries_identifier)):
            df = group[1].drop(timeseries_identifier, axis=1)
            np_df = df.to_numpy()
            t = torch.from_numpy(np_df)
            tensors.append(t)
        self.data = pad_sequence(tensors, batch_first=True, padding_value=0)
        if one_hot:
            self.data = f_one_hot(self.data)

    def __len__(self) -> int:
        return int(len(self.data))

    def __getitem__(self, idx: int) -> Tensor:
        seq: Tensor = self.data[idx]
        return seq.view(seq.size(0), -1)


class AisleDataModule(pl.LightningDataModule):
    """Datamodule for the Aisle dataset."""

    def __init__(
        self,
        batch_size: int = 1,
        lengths: ty.Sequence[ty.Union[int, float]] = [0.8, 0.2],
        num_workers: int = 1,
        pin_memory: bool = True,
        generator: torch.Generator = None,
        **kwargs: ty.Any,
    ) -> None:
        """_summary_
        Args:
            batch_size (int, optional): _description_. Defaults to 1.
            lengths (ty.Sequence[ty.Union[int, float]], optional): _description_. Defaults to [0.8, 0.2].
            num_workers (int, optional): _description_. Defaults to 1.
            pin_memory (bool, optional): _description_. Defaults to True.
        """
        super().__init__()
        self.batch_size = batch_size
        self.lengths = lengths
        self.pin_memory = pin_memory
        self.num_workers = num_workers
        self.dataset = AisleDataset(**kwargs)
        self.train, self.val = random_split(self.dataset, generator=generator, lengths=lengths)

    def train_dataloader(self) -> DataLoader:
        """Training loader."""
        return self._loader(self.train)

    def val_dataloader(self) -> DataLoader:
        """Validation loader."""
        return self._loader(self.val)

    def _loader(self, data: Dataset) -> DataLoader:
        """Utility to avoid code repetition."""
        return DataLoader(
            data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


def df_to_cat(df: pd.DataFrame) -> ty.Tuple[pd.DataFrame, ty.List[str]]:
    """Converts `object` columns to `categorical`."""
    # Find 'object' columns
    to_convert = []
    for col in df.columns:
        logger.trace(f"Checking out {col}: {df[col].dtype}")
        if df[col].dtype == "object":
            to_convert.append(col)
    # Convert them
    logger.trace(f"Converting categorical features {to_convert}")
    for c in to_convert:
        df[c] = df[c].astype("category")
    # Find 'category' columns
    cat_feat = []
    for col in df.columns:
        logger.trace(f"Checking out {col}: {df[col].dtype}")
        if df[col].dtype == "category":
            cat_feat.append(col)
    return df, cat_feat
