import pandas as pd
import numpy as np

from ..synthesizers.base import BaseSynthesizer


class IdentitySynthesizer(BaseSynthesizer):
    """Returns the same data used to train)."""

    def train(self, target_data: pd.DataFrame, *args) -> None:  # type: ignore
        """_summary_
        Args:
            target_data (pd.DataFrame): _description_
        """
        super().train(target_data)  # type: ignore
        self.target_data_ = pd.DataFrame(target_data)

    def generate(self, number_of_subjects: int) -> pd.DataFrame:  # type: ignore
        """_summary_
        Args:
            number_of_subjects (_type_): _description_
        Returns:
            _type_: _description_
        """
        super().generate(self)  # type: ignore

        df_copy = self.target_data_.copy(deep=True)

        unique_ids = df_copy.index.get_level_values(0).unique()
        sampled_ids = np.random.choice(unique_ids, size=number_of_subjects, replace=True)
        grid = pd.DataFrame({"id": sampled_ids}).sort_values("id")
        grid["id_new_"] = range(0, number_of_subjects)  # type: ignore
        df = pd.merge(df_copy.reset_index(), grid, on="id")
        df["id"] = df["id_new_"]
        df = df.drop(["id_new_"], axis=1).sort_values("id")
        df = df.set_index(["id", "sequence_pos"])

        return df
