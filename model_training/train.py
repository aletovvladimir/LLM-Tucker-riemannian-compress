import hydra
import pytorch_lightning as pl

from .utils.utils import get_callbacks, get_data, get_model, get_trainer

# from .utils.utils import PlotLoggerCallback


@hydra.main(config_path="configs", config_name="config")
def main(config):
    pl.seed_everything(config.seed)

    dm = get_data(config)
    model = get_model(config)

    loggers = [
        pl.loggers.MLFlowLogger(
            experiment_name=config.project_name,
            run_name=config.run_name,
            tracking_uri="file:./mlruns",
        )
    ]

    callbacks = get_callbacks(config)
    trainer = get_trainer(config, loggers=loggers, callbacks=callbacks)

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
