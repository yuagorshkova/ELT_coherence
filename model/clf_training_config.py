import torch
import torchmetrics
import pytorch_lightning as pl

from torch import nn


class ClfTrainingConfig(pl.LightningModule):
    def __init__(self):
        super(ClfTrainingConfig, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        
    def _post_init(self):
        self.train_precision = torchmetrics.Precision(task="multiclass", average="macro", num_classes=self.output_size)
        self.train_recall = torchmetrics.Recall(task="multiclass", average="macro", num_classes=self.output_size)
        self.train_f1 = torchmetrics.F1Score(task="multiclass", average="macro", num_classes=self.output_size)
        
        self.eval_precision = torchmetrics.Precision(task="multiclass", average="macro", num_classes=self.output_size)
        self.eval_recall = torchmetrics.Recall(task="multiclass", average="macro", num_classes=self.output_size)
        self.eval_f1 = torchmetrics.F1Score(task="multiclass", average="macro", num_classes=self.output_size)
        

    def training_step(self, batch, batch_idx):
        output = self(**batch)

        total_loss = self.criterion(output["logits"], output["labels"])
        self.log("train/loss", total_loss, prog_bar=True, logger=True)
        
        probs = self.softmax(output["logits"])
        preds = probs.argmax(dim=-1)
        
        self.train_precision(preds, output["labels"])
        self.train_recall(preds, output["labels"])
        self.train_f1(preds, output["labels"])
        self.log("train/precision", self.train_precision, on_step=True, on_epoch=False)
        self.log("train/recall", self.train_recall, on_step=True, on_epoch=False)
        self.log("train/f1-score", self.train_f1, on_step=True, on_epoch=False)
        
        return {
            "loss": total_loss,
        }

    def validation_step(self, batch, batch_idx):
        output = self(**batch)

        total_loss = self.criterion(output["logits"], output["labels"])
        self.log("eval/loss", total_loss, prog_bar=True, logger=True)
        
        probs = self.softmax(output["logits"])
        preds = probs.argmax(dim=-1)
        
        self.eval_precision(preds, output["labels"])
        self.eval_recall(preds, output["labels"])
        self.eval_f1(preds, output["labels"])
        self.log("eval/precision", self.eval_precision, on_step=False, on_epoch=True)
        self.log("eval/recall", self.eval_recall, on_step=False, on_epoch=True)
        self.log("eval/f1-score", self.eval_f1, on_step=False, on_epoch=True)
        return {
            "loss": total_loss,
        }
    
    def test_step(self, batch, batch_idx):
        output = self(**batch)

        total_loss = self.criterion(output["logits"], output["labels"])
        self.log("eval/loss", total_loss, prog_bar=True, logger=True)
        
        probs = self.softmax(output["logits"])
        preds = probs.argmax(dim=-1)
        
        self.eval_precision(preds, output["labels"])
        self.eval_recall(preds, output["labels"])
        self.eval_f1(preds, output["labels"])
        self.log("eval/precision", self.eval_precision)
        self.log("eval/recall", self.eval_recall)
        self.log("eval/f1-score", self.eval_f1)
        return {
            "loss": total_loss,
            "preds":preds,
            "labels":output["labels"],
        }

    def prediction_step(self, batch, batch_idx):
        output = self(**batch)
        probs = self.softmax(output)
        prediction = probs.argmax(dim=-1)
        return {
            "probabilities": probs,
            "prediction": prediction.item(),
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-6)
        return optimizer
    
    def freeze_embedder(self):
        for param in self.encoder.bert.parameters():
            param.requires_grad = False
        
