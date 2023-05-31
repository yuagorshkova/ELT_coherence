def text_tokenize_function(
        examples,
        tokenizer,
):
    encoded_texts = {
        "input_ids": [],
        "attention_mask": [],
        "token_type_ids": [],
        "sentence_offsets": [],
        "label": [],
        "text_id": [],
        
        "sentences":[],
    }
    for text_id, sentences, label in zip(
            examples["text_id"], examples["sentences"], examples["label"]
    ):
        example_input_ids = [tokenizer.cls_token_id]  # one cls token for the text
        example_attention_mask = [1]
        example_token_type_ids = [0]
        example_sentence_offsets = []

        for sentence in sentences:
            tokenized_words = tokenizer(
                sentence,
                return_attention_mask=True,
                add_special_tokens=False,
            )
            first_sentence_token_position = len(example_input_ids)

            for token_input_ids, token_attention_mask, token_type_ids in zip(
                    tokenized_words["input_ids"],
                    tokenized_words["attention_mask"],
                    tokenized_words["token_type_ids"],
            ):
                example_input_ids.extend(token_input_ids)
                example_attention_mask.extend(token_attention_mask)
                example_token_type_ids.extend(token_type_ids)

            last_sentence_token_position = len(example_input_ids) - 1  # inclusive

            example_sentence_offsets.append((
                first_sentence_token_position,
                last_sentence_token_position,
            ))

        encoded_texts["input_ids"].append(example_input_ids)
        encoded_texts["attention_mask"].append(example_attention_mask)
        encoded_texts["token_type_ids"].append(example_token_type_ids)
        encoded_texts["sentence_offsets"].append(example_sentence_offsets)
        encoded_texts["label"].append(label)
        encoded_texts["text_id"].append(text_id)

        encoded_texts["sentences"].append([])

    return encoded_texts


def sentence_tokenize_function(examples, tokenizer):
    encoded_texts = {
        "text_sentences": [],
        "label": [],
        "text_id": [],
    }

    for text_id, sentences, label in zip(
            examples["text_id"], examples["sentences"], examples["label"]
    ):
        example_encoded_sentences = []

        for sentence in sentences:
            encoded_sentence = {
                "input_ids": [],
                "token_type_ids": [],
                "attention_mask": [],
            }
            tokenized_words = tokenizer(
                sentence,
                return_attention_mask=True,
                add_special_tokens=False,
            )

            for token_input_ids, token_attention_mask, token_type_ids in zip(
                    tokenized_words["input_ids"],
                    tokenized_words["attention_mask"],
                    tokenized_words["token_type_ids"],
            ):
                encoded_sentence["input_ids"].extend(token_input_ids)
                encoded_sentence["token_type_ids"].extend(token_type_ids)
                encoded_sentence["attention_mask"].extend(token_attention_mask)

            example_encoded_sentences.append(encoded_sentence)

        encoded_texts["text_sentences"].append(example_encoded_sentences)
        encoded_texts["label"].append(label)
        encoded_texts["text_id"].append(text_id)

    return encoded_texts
