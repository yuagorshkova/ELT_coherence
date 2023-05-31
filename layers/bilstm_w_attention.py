import torch
import pytorch_lightning as pl
import numpy as np

from torch import nn

from typing import Optional, List

class BiLSTM_w_attention(pl.LightningModule):

    """
    Layer that takes encoder output and applies bilstm with attention over it.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: Optional[int] = None,
            lstm_num_layers: int = 1,
    ):
        super().__init__()

        self.input_size = input_size

        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = self.input_size

        self.lstm_num_layers = lstm_num_layers

        self.LSTM = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_num_layers,
            bidirectional=False,
            batch_first=False, 
            # dropout=0.5,
        )

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
            last_hidden_state: torch.Tensor,  # (batch_size, seq_len, embedding_dim)
            labels: Optional[torch.Tensor] = None,
            **kwargs,
    ):
        # apply LSTM
        x = last_hidden_state.transpose(0, 1)  # (seq_len, batch_size, embedding_dim)
        x, _ = self.LSTM(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, embedding_dim)

        attention_weights = torch.tanh(self.attn_linear(x))
        attention_applied = torch.softmax(
            attention_weights.matmul(self.linear_value),
            dim=1,
        )
        # calculate weighted representation using attention weights
        x = torch.sum((attention_applied * x), dim=1) # (batch_size, embedding_dim)

        output = {
            "last_hidden_state": x,
            "labels": labels,
        }
        return output
