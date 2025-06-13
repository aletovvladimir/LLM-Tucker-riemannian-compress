import pytorch_lightning.callbacks as plc
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import Logger
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from typing import List

from .data_class import TDataset
from .model_class import LitModel


def get_model(config: DictConfig) -> pl.LightningModule:
    model = LitModel(config)
    return model


def get_data(config: DictConfig) -> pl.LightningDataModule:
    datamodule = TDataset(config)
    return datamodule


def get_trainer(config: DictConfig, loggers: List[Logger], callbacks: List[Callback]) -> Trainer:
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=loggers)
    return trainer

def get_callbacks(config: DictConfig) -> List[Callback]:
    callbacks = []

    for name in config.callbacks.callbacks:
        if hasattr(plc, name):
            cls = getattr(plc, name)
            kwargs = (
                config.callbacks.callback_kwargs.get(name, {})
                if hasattr(config.callbacks, "callback_kwargs")
                else {}
            )
            callbacks.append(cls(**kwargs))
        else:
            raise ValueError(
                f"Callback '{name}' not found in pytorch_lightning.callbacks"
            )
    return callbacks
