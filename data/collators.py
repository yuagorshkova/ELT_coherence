import torch

from typing import *

from data.data_module import pad_sequence


def text_collate_fn(
        examples: Sequence[dict],
        paddnig_fn: Callable = pad_sequence,
        max_length: int = 512,
):
    input_ids = torch.stack(
        [
            paddnig_fn(f["input_ids"], max_length, 0)
            for f in examples
        ],
        dim=0,
    )
    attention_mask = torch.stack(
        [
            paddnig_fn(f["attention_mask"], max_length, 0)
            for f in examples
        ],
        dim=0,
    )
    token_type_ids = torch.stack(
        [
            paddnig_fn(f["token_type_ids"], max_length, 0)
            for f in examples
        ],
        dim=0,
    )
    labels = torch.tensor([f["labels"] for f in examples])
    sentence_offsets = [f["sentence_offsets"] for f in examples]

    colated_examples = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "sentence_offsets": sentence_offsets,
        "labels": labels,
    }
    return colated_examples


class SentencesCollator():
    def __init__(
        self, 
        paddnig_fn: Callable = pad_sequence,
        max_length: int = 64,
    ):
        self.paddnig_fn = paddnig_fn
        self.max_length = max_length
        
    def __call__(self, examples: Sequence[dict]):
        bs = len(examples)

        input_ids = torch.zeros(bs * 32, self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(bs * 32, self.max_length, dtype=torch.long)
        token_type_ids = torch.zeros(bs * 32, self.max_length, dtype=torch.long)

        batch_input_ids = torch.stack(
            [
                self.paddnig_fn(s["input_ids"], self.max_length, 0)
                for f in examples for s in f["text_sentences"]
            ][:bs * 32],
            dim=0,
        )
        input_ids[:len(batch_input_ids), :] = batch_input_ids

        batch_attention_mask = torch.stack(
            [
                self.paddnig_fn(s["attention_mask"], self.max_length, 0)
                for f in examples for s in f["text_sentences"]
            ][:bs * 32], # pad here
            dim=0,
        )
        attention_mask[:len(batch_input_ids), :] = batch_attention_mask

        batch_token_type_ids = torch.stack(
            [
                self.paddnig_fn(s["token_type_ids"], self.max_length, 0)
                for f in examples for s in f["text_sentences"]
            ][:bs * 32],
            dim=0,
        )
        token_type_ids[:len(batch_input_ids), :] = batch_token_type_ids
        
        

        document_offsets = []
        end = -1
        for f in examples:
            start = end + 1
            end = start + len(f["text_sentences"]) - 1  #inclusive
            if end > bs * 32 - 1:
                document_offsets.append((start, bs * 32 - 1))
                break

            document_offsets.append((start, end))
            
        labels = [f["labels"] for f in examples]
        # labels += [0]*(bs-len(labels))
        labels = torch.tensor(labels[:len(document_offsets)])


        colated_examples = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "document_offsets":document_offsets,
            "labels": labels,
        }
        return colated_examples
