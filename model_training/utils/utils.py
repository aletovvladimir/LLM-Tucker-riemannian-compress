import os

import matplotlib.pyplot as plt
import mlflow
import pytorch_lightning.callbacks as plc
from hydra.utils import instantiate
from pytorch_lightning.loggers import MLFlowLogger

from .data_class import TDataset
from .model_class import LitModel


def get_model(config):
    model = LitModel(config)
    return model


def get_data(config):
    datamodule = TDataset(config)
    return datamodule


def get_trainer(config, loggers, callbacks):
    trainer = instantiate(config.trainer, callbacks=callbacks, logger=loggers)
    return trainer


import pytorch_lightning.callbacks as plc


def get_callbacks(config):
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
