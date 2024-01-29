import pytest
import os
import pyrootutils

root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)

import pandas as pd

from sai.datalab.datasets.loader import load_aisle


@pytest.fixture(scope="session")
def vocab_file() -> str:
    """Vocab file."""
    return str(os.path.join("config", "vocab_file.json"))


@pytest.fixture(scope="session")
def merges_file() -> str:
    """Vocab file."""
    return str(os.path.join("config", "merges_file.json"))


@pytest.fixture(scope="session")
def aisle_visits() -> pd.DataFrame:
    """Whether to test on GPU or not."""
    df = load_aisle().reset_index()
    return df
