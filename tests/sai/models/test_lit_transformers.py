import pytest
from loguru import logger
import sys
import typing as ty

import pytorch_lightning as pl
from transformers import AutoTokenizer
from lightning_transformers.task.nlp.language_modeling import LanguageModelingDataModule

from sai.datasets import AisleLLMDataModule
from sai.models import LLM


def test_llm_on_wiki() -> None:
    """Test lightning-transformers."""
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    logger.info(f"Tokenizer ({type(tokenizer)}): {tokenizer}")
    # LLM
    model = LLM(pretrained_model_name_or_path="gpt2")
    logger.info(f"Model ({type(model)}): {model}")
    # Data
    dm = LanguageModelingDataModule(
        batch_size=1,
        tokenizer=tokenizer,
        dataset_name="wikitext",
        dataset_config_name="wikitext-2-raw-v1",
    )
    # Train
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_steps=16,
        overfit_batches=2,
    )
    trainer.fit(model, dm)
    logger.success("")


def test_llm_on_aisle() -> None:
    """Test lightning-transformers."""
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    logger.info(f"Tokenizer ({type(tokenizer)}): {tokenizer}")
    # LLM
    model = LLM(pretrained_model_name_or_path="gpt2")
    logger.info(f"Model ({type(model)}): {model}")
    # Data
    dm = AisleLLMDataModule(batch_size=1, tokenizer=tokenizer)
    # Train
    trainer = pl.Trainer(
        accelerator="auto",
        devices="auto",
        max_steps=16,
        overfit_batches=2,
    )
    trainer.fit(model, dm)
    logger.success("")


if __name__ == "__main__":
    logger.remove()
    logger.add(sys.stderr, level="DEBUG")
    pytest.main([__file__, "-x", "-s", "--pylint"])
