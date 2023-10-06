import pytest
import pyrootutils, sqlite3, os
import pandas as pd

root = pyrootutils.setup_root(
    search_from=".",
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
    cwd=True,
)


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
    db_connection = sqlite3.connect("Task2/data/aisle_visits.db")
    cursor = db_connection.cursor()
    cursor.execute("SELECT * FROM aisle_visits")
    df = pd.DataFrame(cursor.fetchall(), columns=[column[0] for column in cursor.description])
    cursor.close()
    db_connection.close()
    return df
