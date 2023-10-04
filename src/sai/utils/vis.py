__all__ = ["plot_token_frequencies", "plot_token_positions"]

from loguru import logger
import matplotlib.pyplot as plt
import pandas as pd

from .utils import get_timeseries_frequencies, get_dfs_to_have_common_columns


def plot_token_positions(
    target_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    name: str,
    show: bool = True,
) -> None:
    """Plot token's positions. Compare, for each aisle type, the frequency by which it is visited first, second, etc."""
    logger.debug(f"Looking for {name}")
    df_target = get_timeseries_frequencies(target_data)
    df_syn = get_timeseries_frequencies(synthetic_data)
    df_syn = get_dfs_to_have_common_columns(df_syn, df_target, 0)
    assert len(df_syn.columns) == len(df_target.columns)
    merged_df = pd.merge(
        df_target[name],
        df_syn[name],
        left_index=True,
        right_index=True,
        how="outer",
    )
    merged_df.columns = ["original", "synthetic"]
    merged_df.sort_index().plot(kind="bar", figsize=(20, 6))
    plt.title(f"Visit frequency for {name}")
    plt.xlabel("Position in the sequence [-]")
    plt.ylabel("Frequency [-]")
    if show:
        plt.show()


def plot_token_frequencies(
    target_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    show: bool = True,
) -> None:
    """
    Args:
        target_data (pd.DataFrame): _description_
        synthetic_data (pd.DataFrame): _description_
    """
    # Create pandas Series from the data and remove pad value
    original_series = pd.Series(target_data["aisle"][target_data.aisle != ""])
    synthetic_series = pd.Series(synthetic_data["aisle"][synthetic_data.aisle != ""])
    # Calculate value counts for each category in the original and synthetic data
    original_counts = original_series.value_counts()
    synthetic_counts = synthetic_series.value_counts()
    # Create a DataFrame to compare the counts
    comparison_df = pd.DataFrame(
        {
            "Original": original_counts / sum(original_counts),
            "Synthetic": synthetic_counts / sum(synthetic_counts),
        }
    ).fillna(0)
    # Plot the comparison using a bar chart
    comparison_df.plot(kind="bar", figsize=(20, 6))
    plt.title("Visit frequency")
    plt.xlabel("Aisle type")
    plt.ylabel("Frequency [-]")
    plt.xticks(rotation=90)
    plt.legend()
    if show:
        plt.show()
