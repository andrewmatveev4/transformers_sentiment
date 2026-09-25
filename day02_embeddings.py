from day01_tokenization import tokenize_texts, tokenizer, model_name
from transformers import AutoModel
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity


model = AutoModel.from_pretrained(model_name)
model.eval()


def get_embeddings(texts, tokenizer, model, batch_size=32):
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        tokens = tokenize_texts(batch_texts)

        # Получаем hidden states
        with torch.no_grad():
            outputs = model(**tokens)

        # Извлекаем CLS-токены
        cls_embeddings = outputs.last_hidden_state[:, 0, :]
        all_embeddings.append(cls_embeddings.cpu().numpy())

    # Объединяем все батчи
    return np.vstack(all_embeddings)

def similarity(text1, text2, tokenizer, model):
    emb = get_embeddings([text1, text2], tokenizer, model)
    sim = cosine_similarity(emb[0:1], emb[1:2])[0][0]
    return sim

if __name__ == "__main__":
    # Задача 2: один текст через модель
    text = "This movie was absolutely amazing!"
    tokens = tokenizer(text, return_tensors="pt")
    print(tokens)

    with torch.no_grad():
        outputs = model(**tokens)

    print(outputs.last_hidden_state.shape)

    cls_embedding = outputs.last_hidden_state[:, 0, :]
    print(cls_embedding.shape)
    print(cls_embedding[0][:5])

    texts = [
        "This movie was absolutely amazing!",
        "Terrible movie, waste of time.",
        "Pretty good, I liked it.",
        "Boring and too long."
    ]

    embeddings = get_embeddings(texts, tokenizer, model)
    print(f'Embeddings shape: {embeddings.shape}')
    print(f'Ожидается: (4, 768) для DistilBERT')


    sim1 = similarity("Great movie!", "Amazing film!", tokenizer, model)
    sim2 = similarity("Great movie!", "Terrible film!", tokenizer, model)

    print(f'Сходство похожих: {sim1:.3f}')
    print(f'Сходство разных: {sim2:.3f}')