import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import datasets
from transformers import PreTrainedTokenizerBase

class TDataset(pl.LightningDataModule):
    def __init__(self, data_link:str,
                 tokenizer:PreTrainedTokenizerBase,
                 batch_size:int=56,
                 num_workers:int=8
                 ):
        super().__init__()
        self.link=data_link
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.num_workers=num_workers
     
    def tokenize_func(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True) 
       
    def setup(self, stage=None):
        data = datasets.load_dataset(self.link)
        token_data = data.map(self.tokenize_func, batched=True)
        
        if stage =='fit':
            self.train_dataset = token_data['train']
            self.test_dataset = token_data['test'] 
        else: 
            pass
    
    def prepare_data(self):
        pass
    
    def train_dataloader(self):
        loader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn
        )
        return loader
    
    def test_dataloader(self):
        pass
    
    def collate_fn(self, batch, pad_id=0):
        input_ids = []
        labels = []
        for sample in batch:
            input_ids.append(torch.tensor(sample["input_ids"], dtype=torch.long))
            labels.append(torch.tensor(sample["label"], dtype=torch.long))

        batch = {
            "input_ids": nn.utils.rnn.pad_sequence(input_ids, padding_value=pad_id, batch_first=True),
            # "labels": pad_sequence(labels, padding_value=-100, batch_first=True)
        }
        batch["attention_mask"] = (batch["input_ids"] != pad_id).clone()

        return batch, torch.tensor(labels)