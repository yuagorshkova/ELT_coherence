import torch
import os
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk
from transformers import BertTokenizer

from typing import *

REQUIRED_COLUMNS = [
    "sentences",
    "label",
    "text_id",
]


def pad_sequence(
        seq: Any,
        target_len: int,
        pad_value: Union[List[int], Tuple[int, ...], int] = 0,
        dtype: Any = torch.int,
) -> Any:
    if isinstance(seq, torch.Tensor):
        n = seq.shape[0]
    else:
        n = len(seq)
        seq = torch.tensor(seq, dtype=dtype)
    m = target_len - n
    ret = torch.tensor([pad_value] * m, dtype=seq.dtype)
    ret = torch.cat([seq, ret], dim=0)[:target_len]
    return ret


class TextDataModule(pl.LightningDataModule):
    def __init__(self,
                 data: Optional[Dataset] = None,
                 tokenize_fn: Optional[Callable] = None,
                 collate_fn: Optional[Callable] = None,
                 eval_data: Optional[Dataset] = None,
                 batch_size: int = 32,
                 max_token_len: int = 512,

                 model_name_or_path: Optional[str] = None,
                 
                 tokenized_data: Optional[str] = None,

                 dataloader_num_workers=2,
                 ):
        super().__init__()
        self.data = data
        self.eval_data = eval_data
        self.tokenized_data = tokenized_data
        self.batch_size = batch_size

        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path) if model_name_or_path is not None else None
        self.tokenize_fn = tokenize_fn
        self.collate_fn = collate_fn
        self.max_token_len = max_token_len

        self.dataloader_num_workers = dataloader_num_workers

    def setup(self, stage: str):
        if self.tokenized_data is not None:
            print(f"Loading pretokenized datasets from {self.tokenized_data}")
            self.train_dataset = load_from_disk(self.tokenized_data + "/train")
            self.eval_dataset = load_from_disk(self.tokenized_data + "/eval")
            return
        
        if self.data is not None:
            if isinstance(self.data, pd.DataFrame):
                dataset = Dataset.from_pandas(self.data)
            else:
                dataset = self.data

            removing_columns = list(
                set(dataset.column_names) - set(REQUIRED_COLUMNS)
            )
            print(f"Removing columns in train: {removing_columns}")
            dataset = dataset.remove_columns(removing_columns)

            tokenized_dataset = dataset.map(
                self.tokenize_fn,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                },
                batched=True,
                batch_size=100,
                writer_batch_size=100,
                # remove_columns=["sentences"]
            )

            tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
            print(f"Train dataset columns: {tokenized_dataset.column_names}")
            self.train_dataset = tokenized_dataset
            save_to = os.path.join(os.getcwd(), "tokenized_data", self.tokenize_fn.__name__, "train")
            if not os.path.exists(os.path.join(os.getcwd(), "tokenized_data", self.tokenize_fn.__name__)):
                os.makedirs(os.path.join(os.getcwd(), "tokenized_data", self.tokenize_fn.__name__))
            print(f"Saving train dataset to {save_to}")
            self.train_dataset.save_to_disk(save_to)
        
        
        if self.eval_data is not None:
            if isinstance(self.data, pd.DataFrame):
                dataset = Dataset.from_pandas(self.eval_data)
            else:
                dataset = self.eval_data
                
            removing_columns = list(
                set(dataset.column_names) - set(REQUIRED_COLUMNS)
            )
            print(f"Removing columns in eval: {removing_columns}")
            dataset = dataset.remove_columns(removing_columns)

            tokenized_dataset = dataset.map(
                self.tokenize_fn,
                fn_kwargs={
                    "tokenizer": self.tokenizer,
                },
                batched=True,
                batch_size=100,
                writer_batch_size=100,
                # remove_columns=["sentences"]
            )
            tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
            print(f"Eval dataset columns: {tokenized_dataset.column_names}")

            self.eval_dataset = tokenized_dataset
            save_to = os.path.join(os.getcwd(), "tokenized_data", self.tokenize_fn.__name__, "eval")
            print(f"Saving eval dataset to {save_to}")
            self.eval_dataset.save_to_disk(save_to)
            
                
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.eval_dataset,
            collate_fn=self.collate_fn,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
        )
    
