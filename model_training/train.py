import pytorch_lightning as pl

from transformers import AutoTokenizer

from .model_class import LitModel
from .data_class import TDataset

def main(seed=42, data_link='imdb', model_link='fabriceyhc/bert-base-uncased-imdb',
         rank=(2 ** 8, 2 ** 8, 9), replace_layers_idxs=[2, 3, 4, 5, 6, 7, 8, 9],
         tokenizer=AutoTokenizer,
         batch_size=56, num_workers=8):
    pl.seed_everything(seed=seed)
    
    tokenizer = tokenizer.from_pretrained(model_link)
    dm = TDataset(data_link=data_link, tokenizer=tokenizer, batch_size=batch_size, num_workers=num_workers)
    model = LitModel(model_link=model_link, is_riemann=True, rank=rank, replace_layers_idxs=replace_layers_idxs)
    
    loggers = [pl.loggers.WandbLogger(project='riemann_compression', name='model_training')]
    
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        pl.callbacks.DeviceStatsMonitor(),
        pl.callbacks.RichModelSummary(max_depth=4)
    ]
    
    # callbacks.append(pl.callbacks.ModelCheckpoint(
    #     dirpath=
    # ))
    
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=15,
        logger=loggers,
        callbacks=callbacks
    )
    
    trainer.fit(model, datamodule=dm)
    
if __name__ == '__main__':
    main()