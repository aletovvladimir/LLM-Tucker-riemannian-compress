import datasets
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from omegaconf import DictConfig
import pandas as pd
from sklearn.model_selection import train_test_split

def get_tokenizer(config: DictConfig):
    model_link = config.model_params.model_link

    tokenizer = AutoTokenizer.from_pretrained(model_link)
    return tokenizer


class TDataset(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.cfg = config

        self.data_link = self.cfg.data_params.data_link
        self.batch_size = self.cfg.training_params.batch_size
        self.tokenizer = get_tokenizer(self.cfg)
        self.num_workers = self.cfg.training_params.num_workers

    def tokenize_func(self, examples):
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def setup(self, stage=None):
        data = pd.read_csv(self.data_link)
        data = data.rename(columns={'review' : 'text', 'sentiment':'label'})
        
        train_df, test_df = train_test_split(df, test_size=0.5, random_state=self.config.seed.seed)
        train_dataset = datasets.Dataset.from_pandas(train_df)
        test_dataset = datasets.Dataset.from_pandas(test_df)
        
        dataset = dataset.DatasetDict({
            "train": train_dataset,
            "test": test_dataset,
        })
        
        token_data = dataset.map(self.tokenize_func, batched=True)

        if stage == "fit":
            self.train_dataset = token_data["train"]
            self.val_dataset = token_data["test"]
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
            collate_fn=self.collate_fn,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
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
            "input_ids": nn.utils.rnn.pad_sequence(
                input_ids, padding_value=pad_id, batch_first=True
            ),
            # "labels": pad_sequence(labels, padding_value=-100, batch_first=True)
        }
        batch["attention_mask"] = (batch["input_ids"] != pad_id).clone()

        return batch, torch.tensor(labels)
