from transformers import AutoTokenizer
import torch  # noqa: F401  по заданию; return_tensors="pt" требует torch


model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_texts(texts, max_length=128):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )


def explain_tokenization(text, tokenizer):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)

    print(f"Исходный текст: {text}")
    print(f"Токены: {tokens}")
    print(f"IDs: {ids}")
    print(f"Количество: {len(tokens)}")


if __name__ == "__main__":
    # Задача 2: параметры токенизатора
    print(f"Tokenizer loaded for model: {model_name}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"model_max_length: {tokenizer.model_max_length}")

    # Задача 3: один текст
    text = "This movie was absolutely amazing!"
    tokens = tokenizer(text)
    print(tokens)
    input_ids = tokens["input_ids"]
    print(f"Количество токенов: {len(input_ids)}")
    print(f"Декодировано: {tokenizer.decode(input_ids)}")

    # Задача 4: батч
    texts = [
        "This movie was great!",
        "Terrible movie, waste of time.",
    ]
    batch = tokenize_texts(texts)
    print(f"Shape: {batch['input_ids'].shape}")
    print(f"Attention mask:\n{batch['attention_mask']}")

    # Задача 5: спецтокены
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    single = tokenizer(text, return_tensors="pt")