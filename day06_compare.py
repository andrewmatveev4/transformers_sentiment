import joblib
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from day04_baseline import get_cls_embeddings
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def predict_fine_tuned(texts, model, tokenizer):
    if isinstance(texts, str):
        texts = [texts]

    predictions = []

    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()

        predictions.append({
            "text": text,
            "prediction": pred,
            "probabilities": probs[0].numpy(),
        })

    return predictions

def predict_baseline(texts, model):
    if isinstance(texts, str):
        texts = [texts]

    X = get_cls_embeddings(texts)
    predictions = model.predict(X)
    probs = model.predict_proba(X)

    results = []
    for i, text in enumerate(texts):
        results.append({
            "text": text,
            "prediction": int(predictions[i]),
            "probabilities": probs[i],
        })

    return results

def plot_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["негатив", "позитив"],
        yticklabels=["негатив", "позитив"],
    )
    plt.title(title)
    plt.ylabel("Правда")
    plt.xlabel("Модель сказала")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return cm


if __name__ == "__main__":
    model_ft = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
    tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
    model_ft.eval()

    baseline_model = joblib.load("baseline_model.pkl")

    print(model_ft.classifier)
    print(baseline_model.coef_.shape)
    test_texts = [
        "This movie was absolutely fantastic!",
        "Terrible, waste of my time.",
        "It was okay, nothing special.",
        "Best film I've seen this year!",
        "Boring and too long.",
    ]

    preds_ft = predict_fine_tuned(test_texts, model_ft, tokenizer)
    preds_base = predict_baseline(test_texts, baseline_model)

    for i, text in enumerate(test_texts):
        ft = preds_ft[i]
        base = preds_base[i]
        print(f"\n{text}")
        print(f"  FT:   {ft['prediction']}  позитив {ft['probabilities'][1]:.2f}")
        print(f"  BASE: {base['prediction']}  позитив {base['probabilities'][1]:.2f}")
        print(f"  совпадают: {ft['prediction'] == base['prediction']}")



        df = pd.read_csv("data/sst2_sample.csv")
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    test_texts = test_df["sentence"].tolist()
    test_labels = test_df["label"].tolist()

    y_pred_ft = [p["prediction"] for p in predict_fine_tuned(test_texts, model_ft, tokenizer)]
    y_pred_base = [p["prediction"] for p in predict_baseline(test_texts, baseline_model)]

    f1_ft = f1_score(test_labels, y_pred_ft, average="macro")
    f1_base = f1_score(test_labels, y_pred_base, average="macro")
    acc_ft = accuracy_score(test_labels, y_pred_ft)
    acc_base = accuracy_score(test_labels, y_pred_base)


    cm_ft = plot_confusion_matrix(test_labels, y_pred_ft, "Fine-tuned", "plots/cm_finetuned.png")
    cm_base = plot_confusion_matrix(test_labels, y_pred_base, "Baseline", "plots/cm_baseline.png")
    print(cm_ft)
    print(cm_base)


    errors_ft = int(cm_ft[0, 1] + cm_ft[1, 0])
    errors_base = int(cm_base[0, 1] + cm_base[1, 0])
    improvement = (f1_ft - f1_base) / f1_base * 100

    with open("comparison_results.txt", "w") as f:
        f.write("=== Сравнение моделей (2000 отзывов SST-2) ===\n\n")
        f.write(f"Fine-tuned: F1 {f1_ft:.4f}, Accuracy {acc_ft:.4f}, ошибок {errors_ft}\n")
        f.write(f"Baseline:   F1 {f1_base:.4f}, Accuracy {acc_base:.4f}, ошибок {errors_base}\n\n")
        f.write(f"Улучшение F1: {improvement:.2f}%\n")
        f.write(f"Ошибок меньше на: {errors_base - errors_ft}\n")