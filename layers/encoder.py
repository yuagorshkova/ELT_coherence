import torch
import pytorch_lightning as pl

from transformers import BertModel

from typing import *


class Encoder(pl.LightningModule):
    """
    Encodes text using input ids.

    # encoder = Encoder("bert-base-uncased")
    # encoder_outputs = encoder(**next(iter(train_dataloader)))#["last_hidden_state"]
    # encoder_outputs.keys(), encoder_outputs["labels"]
    """

    def __init__(self, model_path_or_name: str):
        super().__init__()
        self.automatic_optimization = False
        self.bert = BertModel.from_pretrained(
            model_path_or_name,
            output_attentions=True,
            output_hidden_states=True,
        )

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            sentence_offsets: Any = None,
            **kwargs,
    ):
        output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        output["sentence_offsets"] = sentence_offsets
        output["labels"] = labels

        return output
