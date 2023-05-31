import torch
from typing import Optional, Any

from layers import (
    Encoder,
    SentencePooler,
    DocumentEncoder,
    ClassificationHead,
)
from model import ClfTrainingConfig


class HierarchicalModel(ClfTrainingConfig):
    def __init__(
            self,
            model_name: str,
            output_size: int,
            pooling_strategy: str = "attention",
            freeze_embedder: bool = True,
    ):
        super(HierarchicalModel, self).__init__()
        self.model_name = model_name
        self.output_size = output_size
        self.pooling_strategy = pooling_strategy

        self.encoder = Encoder(self.model_name)
        if freeze_embedder:
            self.freeze_embedder()

        self.hidden_size = self.encoder.bert.config.hidden_size

        self.sentence_pooler = SentencePooler(
            pooling_strategy=self.pooling_strategy,
            hidden_size=self.hidden_size,
        )
        self.doc_encoder = DocumentEncoder(self.model_name)
        self.clf = ClassificationHead(self.hidden_size, self.output_size)
        
        self._post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            sentence_offsets: Any = None,
            **kwargs,
    ):
        encoded_tokens = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            sentence_offsets=sentence_offsets,
        )
        encoded_sents = self.sentence_pooler(**encoded_tokens)
        encoded_docs = self.doc_encoder(**encoded_sents)
        logits = self.clf(**encoded_docs)

        return logits
