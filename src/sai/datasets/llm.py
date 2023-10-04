__all__ = ["AisleLLMDataModule", "AisleLLMDataset"]

from loguru import logger
import typing as ty

import torch
from torch.utils.data import Dataset, random_split
from datasets import DatasetDict
from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule
from transformers import AutoTokenizer

from sai.datalab.datasets.loader import load_aisle


class AisleLLMDataset(Dataset):
    """Loads one sequence at a time."""

    def __init__(
        self,
        replace_whitespace: bool = True,
        replace_whitespace_with: str = "-",
        tokenizer: AutoTokenizer = None,
        pretrained_model_name_or_path: str = "gpt2",
    ) -> None:
        """."""
        super().__init__()
        self.replace_whitespace = replace_whitespace
        self.replace_whitespace_with = replace_whitespace_with
        self.df = load_aisle()
        self.tokenizer = (
            AutoTokenizer.from_pretrained(pretrained_model_name_or_path) if tokenizer is None else tokenizer
        )

    def __len__(self) -> int:
        """Length of the dataset."""
        return int(len(self.df.reset_index()["id"].unique()))

    def __getitem__(self, idx: int) -> ty.Dict[str, ty.Any]:
        """Return one sequence."""
        tmp = self.df.reset_index()
        seq = tmp.loc[tmp["id"] == 0]
        s = seq["aisle"].to_list()
        if self.replace_whitespace:
            s = [s.replace(" ", self.replace_whitespace_with) for s in s]
        logger.trace(f"Sequence: {s}")
        result_string = " ".join(s).strip()
        logger.trace(f"Sequence: {result_string}")
        if self.tokenizer is None:
            return {"text": result_string}
        tok = self.tokenizer(result_string)
        logger.trace(f"Tokens ({type(tok)}): {tok}")
        if "labels" not in tok:
            tok["labels"] = tok["input_ids"]
        return tok  # type: ignore


class AisleLLMDataModule(LanguageModelingDataModule):
    """Overrides `LanguageModelingDataModule`."""

    def __init__(
        self,
        *args: ty.Any,
        split: ty.Sequence[float] = [0.7, 0.2, 0.1],
        tokenizer: torch.nn.Module = AutoTokenizer.from_pretrained("gpt2"),
        **kwargs: ty.Any,
    ) -> None:
        """
        Args:
            split (ty.Sequence[float], optional): _description_. Defaults to [0.7, 0.2, 0.1].
        """
        super().__init__(*args, tokenizer=tokenizer, **kwargs)
        self.split = split
        self.ds: DatasetDict

    def load_dataset(self) -> DatasetDict:
        """Override."""
        dataset = AisleLLMDataset()
        train, val, test = random_split(dataset, self.split)
        return DatasetDict({"test": test, "train": train, "validation": val})

    def setup(self, stage: str = None) -> None:
        """Set up."""
        logger.trace(stage)
        dataset = self.load_dataset()
        self.ds = dataset
