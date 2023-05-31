import argparse
import os
import torch
import pandas as pd
import pytorch_lightning as pl

from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from typing import Optional
from datasets import Dataset, load_from_disk, concatenate_datasets

from data import TextDataModule, text_collate_fn, text_tokenize_function
from model import HierarchicalModel


def main(args):
    pl.seed_everything(1000)

    train_datasets = []
    
    if args.tokenized_datasets is not None:
        data_module = TextDataModule(
            tokenized_data=args.tokenized_datasets,
            collate_fn=text_collate_fn, 
            dataloader_num_workers=8,
            batch_size=8,
        )
        
    else:
        if args.gcdc is not None:
            if os.path.exists(args.gcdc):
                gcdc_data = load_from_disk(args.gcdc)
                train_datasets.append(gcdc_data)
            else:
                raise Exception(f"Incorrect path to GCDC dataset: {args.gcdc}")

        if args.ellipse is not None:
            if os.path.exists(args.ellipse):
                ellipse_train_data = load_from_disk(args.ellipse)
                train_datasets.append(ellipse_train_data)
            else:
                raise Exception(f"Incorrect path to ELLIPSE dataset: {args.ellipse}")

        if args.eval is not None:
            if os.path.exists(args.eval):
                ellipse_test_data = load_from_disk(args.eval)
            else:
                raise Exception(f"Incorrect path to eval ELLIPSE dataset: {args.eval}")
    
        train_data = concatenate_datasets(train_datasets)
        
        data_module = TextDataModule(
            train_data, 
            text_tokenize_function, 
            text_collate_fn, 
            eval_data=ellipse_test_data,
            model_name_or_path="/home/jovyan/vkr/weights/bert-base-cased",
            batch_size=8,
            dataloader_num_workers=8,
        )

    model = HierarchicalModel(
        model_name="/home/jovyan/vkr/weights/bert-base-cased", 
        output_size=3,
    )

    torch.cuda.empty_cache()
    
    csv_logger = CSVLogger("/home/jovyan/vkr/artifacts/logs", name=args.run_id)
    wandb_logger = WandbLogger(project="tc", name=args.run_id)
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k=3,
        monitor="eval/loss",
        mode="min",
        dirpath=f"/home/jovyan/vkr/artifacts/cpoints/{args.run_id}",
        filename="model_{epoch:02d}-{global_step}",
        save_on_train_epoch_end=True,
    )
    
    trainer = pl.Trainer(
        logger=[
            csv_logger, wandb_logger,
        ],
        callbacks = [
            checkpoint_callback
        ],
        accelerator="gpu",
        devices=[args.gpu],
        max_epochs=1000,
        enable_progress_bar=True,
    )
    
    trainer.fit(model, data_module)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--gpu", type=int, default=0)
    argparser.add_argument("--run_id", type=str, default="run")
    argparser.add_argument("--gcdc", type=str, default=None, help="Path to GCDC train data")
    argparser.add_argument("--ellipse", type=str, default=None, help="Path to ELLIPSE train data")
    argparser.add_argument("--eval", type=str, default=None, help="Path to eval train data")
    argparser.add_argument("--tokenized_datasets", type=str, default=None, help="Path to arrow datasets")
    
    args = argparser.parse_args()
    
    main(args)