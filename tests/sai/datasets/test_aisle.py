import pytest
from loguru import logger
import sys
import typing as ty
from transformers import AutoTokenizer
import lightning.pytorch as pl
from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule
import pytorch_forecasting as pyf

from sai.datasets import AisleLLMDataModule
from sai.datalab.datasets.loader import load_aisle


def test_data() -> None:
    """Test correct shape. And playing around with pytorch-forecasting, using `TimeSeriesDataSet`."""
    # Test shapes
    df = load_aisle(pad=False).reset_index()
    logger.info(df.shape)
    logger.info(f"IDs ({len(df['id'].unique())}): {(df['id'].unique())}")
    logger.info(f"Seq ({len(df['sequence_pos'].unique())}): {(df['sequence_pos'].unique())}")
    assert len(df["sequence_pos"].unique()) == 45
    # Play with pytorch-forecasting
    dataset = pyf.TimeSeriesDataSet(
        data=df,
        time_idx="sequence_pos",
        group_ids=["id"],
        target="aisle",
        time_varying_known_categoricals=["aisle"],
        min_encoder_length=2,
    )
    logger.info(f"Dataset: {dataset}")
    loader = dataset.to_dataloader(batch_size=1)
    for X, y in loader:
        logger.info(f"X: {X}")
        logger.info(f"y: {y}")
        break


def test_aislellmdataset() -> None:
    """Test dataset."""
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    logger.info(f"Tokenizer ({type(tokenizer)}): {tokenizer}")
    # Test
    dm = AisleLLMDataModule(batch_size=1, tokenizer=tokenizer)
    dm.prepare_data()
    dm.setup()
    for i, s in enumerate(dm.ds):  # type: ignore
        logger.info(f"Sample ({type(s)}): {s}")
        if i > 1:
            break
    _show_batch(dm)


def _show_batch(dm: pl.LightningDataModule) -> None:
    for batch in dm.train_dataloader():
        names = list(batch.keys())  # type: ignore
        assert "input_ids" in names
        assert "labels" in names
        assert "attention_mask" in names
        assert batch["input_ids"].dim() == batch["labels"].dim() == batch["attention_mask"].dim() == 2  # type: ignore
        for key, val in batch.items():  # type: ignore
            logger.info(f"{key}: {val.size()}")  # type: ignore
        break


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="TRACE")
    pytest.main([__file__, "-x", "-s", "--pylint"])
