import torch
from typing import Optional, Any

from layers import Encoder, BiLSTM_w_attention, ClassificationHead
from model import ClfTrainingConfig

# up to bs * 64 sents in batch (collator)
# up to 64 sents per doc (here)
class STLModel(ClfTrainingConfig):
    def __init__(
            self,
            model_name: str,
            output_size: int,
            hidden_size: Optional[int] = None,
            lstm_num_layers: int = 1,
            return_sentence_encodings=False,
            freeze_embedder=True,
    ):
        super(STLModel, self).__init__()
        self.model_name = model_name
        self.encoder = Encoder(self.model_name)
        if freeze_embedder:
            self.freeze_embedder()
        
        self.input_size = self.encoder.bert.config.hidden_size
        self.output_size = output_size
        if hidden_size is not None:
            self.hidden_size = hidden_size
        else:
            self.hidden_size = self.input_size
        self.lstm_num_layers = lstm_num_layers

        self.sentence_encoder = BiLSTM_w_attention(
            self.input_size, self.hidden_size, self.lstm_num_layers
        )
        self.document_encoder = BiLSTM_w_attention(
            self.input_size, self.hidden_size, self.lstm_num_layers
        )
        self.clf = ClassificationHead(self.hidden_size, self.output_size)
        
        self.return_sentence_encodings = return_sentence_encodings
        
        self._post_init()

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            sentence_offsets: Any = None,
            document_offsets: Any = None,
            **kwargs,
    ):
        encoded_tokens = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            sentence_offsets=sentence_offsets,
        )
        
        sentence_representations = self.sentence_encoder(**encoded_tokens)
        
        sents_by_docs = []
        sents_by_docs_wo_padding = []
        for start, end in document_offsets:
            sents_by_docs.append(
                torch.nn.functional.pad(sentence_representations["last_hidden_state"][start : end+1, :], (0, 0, 0, 32 - end + start -1))
            )
            if self.return_sentence_encodings:
                sents_by_docs_wo_padding.append(sentence_representations["last_hidden_state"][start : end+1, :])
                
        sents_by_docs = torch.stack(sents_by_docs)
        sentence_representations["last_hidden_state"] = sents_by_docs
        
        document_representations = self.document_encoder(**sentence_representations)
        logits = self.clf(**document_representations)
        
        if self.return_sentence_encodings:
            logits["document_offsets"] = document_offsets
            logits["sentence_representations"] = sents_by_docs_wo_padding
        return logits
