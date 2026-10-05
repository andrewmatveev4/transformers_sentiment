from day01_tokenization import tokenize_texts
from day02_embeddings import get_embeddings, tokenizer, model
import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

def get_cls_embeddings(texts, batch_size=32):
    return get_embeddings(texts, tokenizer, model, batch_size)


if __name__ == "__main__":
    test_texts = [
        "This movie was great!",
        "Terrible movie, waste of time.",
        "Pretty good, I liked it.",
    ]

    # Задача 1.4: проверка токенизации
    print(tokenize_texts(test_texts))

    # Задача 2.5: проверка эмбеддингов
    embeddings = get_cls_embeddings(test_texts)
    print(embeddings.shape)

    df = pd.read_csv("data/sst2_sample.csv")
    texts = df["sentence"].tolist()
    y = df["label"].values

    emb_path = "data/sst2_embeddings.npy"
    if os.path.exists(emb_path):
        X = np.load(emb_path)
    else:
        X = get_cls_embeddings(texts)
        np.save(emb_path, X)
    print(X.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(X_train.shape, X_test.shape)


    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro F1: {f1:.4f}")

    with open("baseline_results.txt", "w") as f:
        f.write(f"Macro F1: {f1:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))