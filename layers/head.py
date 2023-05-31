import torch
import pytorch_lightning as pl

from torch import nn

class ClassificationHead(pl.LightningModule):
    """
    Gets cls token and predicts.

        # clf = ClassificationHead(768, 3)
        # res = clf.forward(**doc_emb)
        # res
    """

    def __init__(
            self,
            hidden_size: int,
            output_size: int,
            dropout: float = 0.2,
    ):
        super(ClassificationHead, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(self.hidden_size, self.output_size, bias=True)

    def forward(
            self,
            last_hidden_state: torch.Tensor = None,
            labels: torch.Tensor = None,
            **kwargs,

    ):
        if last_hidden_state is not None and last_hidden_state.ndim > 2:
            x = last_hidden_state[:, 0, :]  # take cls
        elif last_hidden_state is not None:
            x = last_hidden_state
        else:
            raise ValueError
            
        x = self.dropout(x)
        x = self.linear(x)

        output = {
            "logits": x,
            "labels": labels,
        }

        return output
