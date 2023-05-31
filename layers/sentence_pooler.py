import torch
import pytorch_lightning as pl
import numpy as np

from torch import nn

from typing import *

class SentencePooler(pl.LightningModule):
    """
    Gets token embeddings from a transformer and makes them into sentence embeddings.

        # sent_pooler = SentencePooler(
        #     pooling_strategy="attention",
        #     hidden_size=encoder.bert.config.hidden_size,
        # )
        # sent_embeddings = sent_pooler.forward(**encoder_outputs)
        # sent_embeddings

    """

    def __init__(
            self,
            pooling_strategy: str,
            hidden_size: int,
    ):
        super(SentencePooler, self).__init__()
        self.pooling_strategy = pooling_strategy
        self.hidden_size = hidden_size

        if self.pooling_strategy == "attention":
            self.attn_linear = nn.Linear(
                self.hidden_size, self.hidden_size
            )
            self.linear_value = nn.Parameter(
                nn.init.xavier_uniform_(
                    torch.FloatTensor(self.hidden_size, 1),
                    gain=np.sqrt(2.0),
                ),
                requires_grad=True,
            )


    def forward(
            self,
            last_hidden_state: torch.Tensor,
            sentence_offsets: List[tuple],
            labels: torch.Tensor,
            **kwargs,
    ):

        if self.pooling_strategy == "max":
            return
        elif self.pooling_strategy == "attention":
            pooler_output, attn_mask, token_type_ids = self.attention_pooler(
                last_hidden_state,
                sentence_offsets
            )

        pooler_output = {
            "sent_pooler_output": pooler_output,
            "sent_attention_mask": attn_mask,
            "token_type_ids": token_type_ids,
            "labels": labels,
        }
        return pooler_output

    def attention_pooler(
            self,
            last_hidden_state: torch.Tensor,
            sentence_offsets: List[tuple],
    ):
        batch_size = last_hidden_state.size(0)
        max_sent_count = max(
            len(sentences) for sentences in sentence_offsets
        ) + 1  # to account for cls

        device = last_hidden_state.device

        sentence_embeddings = torch.zeros(
            batch_size, max_sent_count, self.hidden_size,
        ).to(device)
        sentence_attention_mask = torch.zeros(
            batch_size, max_sent_count, dtype=torch.long,
        ).to(device)
        token_type_ids = torch.zeros(
            batch_size, max_sent_count, dtype=torch.long,
        ).to(device)

        # isn't possible to do vector operations as we are using sentence offsets
        for batch_idx in range(batch_size):
            text_sentence_offsets = sentence_offsets[batch_idx]
            inputs = last_hidden_state[batch_idx]

            sentence_embeddings[batch_idx, 0] = inputs[batch_idx, 0]  # cls
            sentence_attention_mask[batch_idx, 0] = 1

            for sent_i, (start, end) in enumerate(text_sentence_offsets):
                attention_weights = torch.tanh(
                    self.attn_linear(inputs[start: end + 1])
                )

                attention_applied = torch.softmax(
                    attention_weights.mm(self.linear_value),
                    dim=0,
                )
                # calculate weighted representation using attention weights
                sentence_representation = torch.sum(
                    (attention_applied * inputs[start: end + 1]),
                    dim=0,
                )

                sentence_embeddings[batch_idx, sent_i + 1] = sentence_representation
                sentence_attention_mask[batch_idx, sent_i + 1] = 1

        return sentence_embeddings, sentence_attention_mask, token_type_ids
