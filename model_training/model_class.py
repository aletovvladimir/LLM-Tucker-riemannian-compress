import torch.nn
import pytorch_lightning as pl

from model_compression import RiemannModel, TuckerLinear, TRAdam
from transformers import AutoModelForSequenceClassification

class LitModel(pl.LightningModule):
    
    def tucker_replace_attention(self, model):
        bert = next(model.children())
        bert_layers = list(list(bert.children())[1].children())[0]
        params = []
        for layer_idx, bert_layer in enumerate(bert_layers):
            if self.replace_layers_idxs is not None and layer_idx not in self.replace_layers_idxs:
                continue
            bert_attention = list(bert_layer.children())[0]
            bert_self = list(bert_attention.children())[0]
            bert_self_output = list(bert_attention.children())[1]
            layers = [bert_self.query, bert_self.key, bert_self.value, bert_self_output.dense]
            new_layers = []
            for layer in layers:
                new_layer = TuckerLinear(768, 768, layer, self.rank, [[2 ** 4, 2 ** 4, 3, 2 ** 4, 2 ** 4, 3],
                                                                [2 ** 8, 2 ** 8, 9]])
                params.append({
                    "params": new_layer.riemann_parameters(),
                    "rank": self.rank
                })
                new_layers.append(new_layer)
            bert_self.query, bert_self.key, bert_self.value, bert_self_output.dense = new_layers
        model = RiemannModel(model)
        return model, params
    
    def __init__(self, model_link, is_riemann, lr=5e-4,  **riemann_kwargs):
        super().__init__()

        self.is_riemann = is_riemann
        self.lr = lr
        if self.is_riemann:
            self.rank = riemann_kwargs.get('rank', ((2 ** 8, 2 ** 8, 9)))
            self.replace_layers_idxs = riemann_kwargs.get('replace_layers_idxs', [2, 3, 4, 5, 6, 7, 8, 9])
            self.model, self.params = self.tucker_replace_attention(AutoModelForSequenceClassification.from_pretrained(model_link))
            self.freeze_params()    
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_link)
            
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    def training_step(self, batch, batch_idx, dataloader_idx=0):
        batch_in, targets = batch
        input_ids = batch_in['input_ids']
        attention_mask = batch_in['attention_mask']
        preds = self(input_ids, attention_mask=attention_mask, labels=targets)
        loss = preds.loss
        self.log('train_loss', loss, prog_bar=True, on_epoch=True,)
        return loss
    
    def test_step(self, *args, **kwargs):
        pass
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        batch_in, targets = batch
        input_ids = batch_in['input_ids']
        attention_mask = batch_in['attention_mask']
        preds = self(input_ids, attention_mask=attention_mask, labels=targets)
        loss = preds.loss
        acc = (preds.logits.argmax(dim=1) == targets).float().mean()
        self.log('test_loss', loss, prog_bar=True, on_epoch=True,)
        self.log('test_accuracy', acc, prog_bar=True, on_epoch=True,)
        return {'test_loss' : loss,
                'test_accuracy': acc}
        
    def freeze_params(self):
        for p in self.model.regular_parameters():
            p.requires_grad_(False) 
        
    
    def configure_optimizers(self):
        # if self.is_riemann:
        #     return TRAdam(self.params)
        print(list(self.params))
        optim = torch.optim.Adam(self.params, lr=self.lr) 
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.7,)
        return {
        "optimizer": optim,
        "lr_scheduler": scheduler
    }
            
        