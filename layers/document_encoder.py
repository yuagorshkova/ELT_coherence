import torch
import pytorch_lightning as pl

from transformers import BertModel, AutoConfig, AutoModel


class DocumentEncoder(pl.LightningModule):
    """
    Gets sentence embeddings and passes them through a transformer.
        # doc_encoder = DocumentEncoder("bert-base-uncased")
        # doc_emb = doc_encoder.forward(**sent_embeddings)
        # doc_emb.keys()
        # doc_emb["last_hidden_state"].size()
    """

    def __init__(self, model_path_or_name):
        super().__init__()
        self.automatic_optimization = False
        # self.bert = BertModel.from_pretrained(
        #     model_path_or_name,
        #     output_attentions=True,
        #     output_hidden_states=True,
        # )
        self.bert = AutoModel.from_config(
            AutoConfig.from_pretrained(
                model_path_or_name,
                add_pooling_layer=False,
                output_attentions=True,
                output_hidden_states=True,
            )
        )

    def forward(
            self,
            sent_pooler_output: torch.Tensor = None,
            sent_attention_mask: torch.Tensor = None,
            token_type_ids: torch.Tensor = None,
            labels: torch.Tensor = None,
            **kwargs,
    ):
        if "sent_pooler_output" is not None:
            output = self.bert(
                inputs_embeds=sent_pooler_output,
                attention_mask=sent_attention_mask,
                token_type_ids=token_type_ids,

            )
        else:
            raise NotImplementedError

        output["labels"] = labels
        return output


class TextualEntailmentHead(pl.LightningModule):
    """
    Maybe: for each pair of sentences take cls and classify them
    But if i figure out how to do it from sentence embeddins it will be faster
    """

    def __init__(self):
        pass

    def forward(self, inputs):
        pass
