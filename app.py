import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_ft = AutoModelForSequenceClassification.from_pretrained("./fine_tuned_model")
tokenizer = AutoTokenizer.from_pretrained("./fine_tuned_model")
model_ft.eval()

LABELS = {0: "Negative", 1: "Positive"}
THRESHOLD = 0.7


def predict_sentiment(text):
    if not text.strip():
        return "Введите текст"

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model_ft(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0]
    pred = int(torch.argmax(probs))
    confidence = float(probs[pred])

    if confidence >= THRESHOLD:
        verdict = LABELS[pred]
    else:
        verdict = f"Не уверена (скорее {LABELS[pred]})"

    result = f"Prediction: {verdict}\n\nProbabilities:\n"
    for i, label in LABELS.items():
        result += f"{label}: {probs[i] * 100:.2f}%\n"
    return result


demo = gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=3, placeholder="Введите отзыв на английском..."),
    outputs=gr.Textbox(label="Результат"),
    title="Sentiment Analysis с DistilBERT",
    description="Дообученная на SST-2 модель определяет тональность отзыва",
)

if __name__ == "__main__":
    demo.launch()