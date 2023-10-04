__all__ = [
    "setup_from_cfg",
    "DEFAULT_TRAINER_KWARGS",
    "get_stage",
    "SUPPORTED_STAGES",
]

from loguru import logger
import typing as ty

import hydra
from omegaconf.errors import UnsupportedInterpolationType
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers.logger import Logger


DEFAULT_TRAINER_KWARGS = dict(
    max_epochs=2,
    logger=False,
    num_sanity_val_steps=2,
    accelerator="auto",
    devices=1,
    enable_checkpointing=False,
)

SUPPORTED_STAGES = ("fit", "validate", "test", "assess", "attack", "debug")


def get_stage(cfg: DictConfig) -> str:
    """Perform sanity checking."""
    assert "stage" in cfg.keys(), f"You must provide a stage: {SUPPORTED_STAGES}"
    stage: str = cfg.get("stage")
    stage = stage.lower().strip()
    assert (
        stage in SUPPORTED_STAGES
    ), f"Unsupported stage: please choose one of the following {SUPPORTED_STAGES}"
    return stage


def setup_from_cfg(
    cfg: DictConfig,
    setup_trainer: bool = True,
) -> ty.Tuple[pl.LightningModule, pl.LightningDataModule, ty.Optional[pl.Trainer], str,]:
    """Helper function to instantiate model, data and trainer."""
    logger.info(f"Setting up with configuration:\n{OmegaConf.to_yaml(cfg)}")
    # get stage
    stage = get_stage(cfg)
    # create model
    pl_model: pl.LightningModule
    model_class: ty.Type[pl.LightningModule]
    if hasattr(cfg, "ckpt_path") and cfg.ckpt_path is not None:
        model_class = hydra.utils.get_class(cfg.model._target_)
        pl_model = model_class.load_from_checkpoint(cfg.ckpt_path)
        logger.info(f"Successfully loaded model from checkpoint {cfg.ckpt_path}")
    elif hasattr(cfg, "load_best_on_dataset") and cfg.load_best_on_dataset is not None:
        best_cfg: dict = hydra.utils.instantiate(cfg.load_best_on_dataset, _convert_="all")
        model_class = hydra.utils.get_class(cfg.model._target_)
        pl_model = model_class.load_best_on_dataset(**best_cfg)
        name = best_cfg.get("name", None)
        logger.info(f"Successfully loaded model {pl_model.__class__.__name__}, best on {name}")
    else:
        pl_model = hydra.utils.instantiate(cfg.model, _convert_="all")
    logger.info(f"Model: {pl_model.__class__.__name__}")
    # create datamodule
    pl_datamodule: pl.LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    logger.info(f"Data: {pl_datamodule.__class__.__name__}")
    # NOTE: the 'logger' and 'callbacks' keys may be under cfg or under cfg.trainer
    pl_trainer = None
    if setup_trainer:
        # create loggers: they may be undefined
        pl_loggers = _get_loggers(cfg)
        # create callbacks: they may be undefined
        pl_callbacks = _get_callbacks(cfg)
        # set trainer.logger and trainer.callbacks with the correct values
        trainer: dict = OmegaConf.to_container(cfg.trainer)  # type: ignore
        trainer.pop("logger", None)
        trainer.pop("callbacks", None)
        trainer.pop("default_root_dir", None)
        # create trainer
        logger.info(f"Trainer with parameters:\n{trainer}")
        logger.info(f"Callbacks: {[type(x) for x in pl_callbacks]}")
        try:
            default_root_dir = cfg.paths.output_dir
        except UnsupportedInterpolationType:
            default_root_dir = None
        pl_trainer = pl.Trainer(
            default_root_dir=default_root_dir,
            logger=pl_loggers if len(pl_loggers) > 0 else True,
            callbacks=pl_callbacks if len(pl_callbacks) > 0 else None,  # type: ignore
            **trainer,
        )
    # return
    return pl_model, pl_datamodule, pl_trainer, stage


def _get_callbacks(cfg: DictConfig) -> ty.List[pl.Callback]:
    """Extract callbacks from the given configuration. Callbacks may be either under `cfg` or under `cfg.trainer`, or under both. We will extract them all."""
    pl_callbacks: list = []
    if "callbacks" in cfg:
        out = _get_cb(cfg)
        pl_callbacks += out
    if "callbacks" in cfg.trainer:
        out = _get_cb(cfg.trainer)
        pl_callbacks += out
    # clean up objects that are not callbacks
    idx_to_pop = []
    for i, cb in enumerate(pl_callbacks):
        if not isinstance(cb, pl.Callback):
            idx_to_pop.append(i)
    for i in reversed(idx_to_pop):
        pl_callbacks.pop(i)
    for cb in pl_callbacks:
        assert isinstance(cb, pl.Callback)
    # return
    return pl_callbacks


def _get_cb(cfg: DictConfig) -> ty.List[pl.Callback]:
    """Helper."""
    out: ty.List[pl.Callback] = []
    config: DictConfig = cfg.callbacks
    logger.debug(config)
    cb = hydra.utils.instantiate(config)
    if isinstance(cb, DictConfig):
        out = list(cb.values())
        assert isinstance(out, list)
        assert isinstance(out[0], pl.Callback)
        return out
    if isinstance(cb, (OmegaConf, ListConfig)):
        out = OmegaConf.to_container(cb)  # type: ignore
    elif isinstance(cb, pl.Callback):
        out = [cb]
    else:
        raise ValueError()
    return out


def _get_loggers(cfg: DictConfig) -> ty.Sequence[Logger]:
    """Extract loggers from the configuration."""
    pl_loggers: list = []
    pl_loggers += instantiate_loggers(cfg.get("logger"))
    return pl_loggers


def instantiate_loggers(
    logger_cfg: DictConfig,
) -> ty.List[Logger]:
    """Instantiates loggers from config."""
    loggers: ty.List[Logger] = []
    if not logger_cfg:
        logger.warning(f"Logger config is empty: {logger_cfg}")
        return loggers
    if not isinstance(logger_cfg, DictConfig):
        raise TypeError("Logger config must be a DictConfig!")
    for _, lg_conf in logger_cfg.items():
        if isinstance(lg_conf, DictConfig) and "_target_" in lg_conf:
            logger.info(f"Instantiating logger <{lg_conf._target_}> with config: {lg_conf}")
            lg = hydra.utils.instantiate(lg_conf)
            loggers.append(lg)
    return loggers
