# pylint: disable=unused-variable
# pylint: disable=unused-import
# pylint: disable=abstract-class-instantiated
# pylint: disable=broad-except
import warnings

warnings.filterwarnings("ignore")

import typing as ty
from loguru import logger
import pyrootutils, os
import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl

from sai.pipeline.utils import setup_from_cfg
from sai.utils import get_metric


ROOT = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)


@hydra.main(
    config_path=os.path.join(ROOT, "configs"),
    config_name="main",
)
def main(cfg: DictConfig = None) -> ty.Optional[float]:
    """Train model. You can pass a different configuration from the command line as follows:
    >>> python main.py --config-name <name>
    """
    assert cfg is not None
    pl_model, pl_datamodule, pl_trainer, stage = setup_from_cfg(cfg)
    # Return if no Trainer
    if pl_trainer is None:
        return 0.0
    # Select method based on `stage`
    if stage in ["fit", "train"]:
        pl_trainer.fit(pl_model, pl_datamodule)
    elif stage in ["val", "validate", "validation"]:
        pl_trainer.validate(pl_model, pl_datamodule)
    elif stage in ["test", "testing"]:
        pl_trainer.test(pl_model, pl_datamodule)
    else:
        pl_trainer.predict(pl_model, pl_datamodule)
    # Get metric to optmizer if HPO
    out = get_metric(pl_trainer, cfg)
    return out


if __name__ == "__main__":
    """You can pass a different configuration from the command line as follows:
    >>> python main.py --config-name <name>
    """
    main()
