import os

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from day06_compare import predict_fine_tuned


if __name__ == "__main__":
    preds_path = "data/test_predictions.csv"

    if os.path.exists(preds_path):
        df_test = pd.read_csv(preds_path)
    else:
        model_ft = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
        tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
        model_ft.eval()

        df = pd.read_csv("data/sst2_sample.csv")
        _, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
        texts = test_df["sentence"].tolist()
        preds = predict_fine_tuned(texts, model_ft, tokenizer)

        df_test = pd.DataFrame({
            "text": texts,
            "true_label": test_df["label"].tolist(),
            "pred_label": [p["prediction"] for p in preds],
            "prob_pos": [p["probabilities"][1] for p in preds],
        })
        df_test.to_csv(preds_path, index=False)

    errors = df_test[df_test["true_label"] != df_test["pred_label"]]
    fp = errors[(errors["pred_label"] == 1) & (errors["true_label"] == 0)]
    fn = errors[(errors["pred_label"] == 0) & (errors["true_label"] == 1)]

    print(f"Всего ошибок: {len(errors)}")
    print(f"False Positives: {len(fp)}")
    print(f"False Negatives: {len(fn)}")


    confident_fp = fp.sort_values("prob_pos", ascending=False).head(10)
    confident_fn = fn.sort_values("prob_pos", ascending=True).head(10)

    print("\n=== FP: уверенно сказала ПОЗИТИВ, а был негатив ===")
    for _, row in confident_fp.iterrows():
        print(f"{row['prob_pos']:.3f} | {row['text']}")

    print("\n=== FN: уверенно сказала НЕГАТИВ, а был позитив ===")
    for _, row in confident_fn.iterrows():
        print(f"{row['prob_pos']:.3f} | {row['text']}")

    print(f"\nСредняя длина ошибок: {errors['text'].str.len().mean():.0f} символов")
    print(f"Средняя длина всех:   {df_test['text'].str.len().mean():.0f} символов")


    observations = [
        "Сарказм и ирония: хвалебные слова в издевательской фразе модель принимает за похвалу ('worthy of the gong').",
        "Отрицание в идиомах: 'can't recommend it enough' это похвала, но модель видит 'can't' и ставит негатив.",
        "Отдельные слова вместо смысла: 'laughable' считается позитивом из-за 'laugh'.",
        "Опечатки и редкие слова режутся токенизатором на бессмысленные куски ('revigorates', 'venality').",
        "Часть ошибок не вина модели: SST-2 состоит из обрывков фраз без контекста, некоторые метки спорны.",
        "Ошибки уверенные (0.999 / 0.002): порог уверенности их не отсеет, модель плохо откалибрована.",
        "Длина текста на ошибки не влияет: 53 символа у ошибок против 54 в среднем.",
    ]

    with open("error_analysis.txt", "w") as f:
        f.write("=== АНАЛИЗ ОШИБОК (fine-tuned, 2000 отзывов SST-2) ===\n\n")
        f.write(f"Всего ошибок: {len(errors)}\n")
        f.write(f"False Positives: {len(fp)}\n")
        f.write(f"False Negatives: {len(fn)}\n\n")

        f.write("=== САМЫЕ УВЕРЕННЫЕ FALSE POSITIVES ===\n")
        for _, row in confident_fp.iterrows():
            f.write(f"{row['prob_pos']:.3f} | {row['text']}\n")

        f.write("\n=== САМЫЕ УВЕРЕННЫЕ FALSE NEGATIVES ===\n")
        for _, row in confident_fn.iterrows():
            f.write(f"{row['prob_pos']:.3f} | {row['text']}\n")

        f.write("\n=== НАБЛЮДЕНИЯ ===\n")
        for line in observations:
            f.write(f"- {line}\n")