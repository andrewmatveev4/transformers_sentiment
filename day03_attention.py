from day01_tokenization import tokenizer, model_name
from transformers import AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import torch


model = AutoModel.from_pretrained(model_name, output_attentions=True)
model.eval()


def visualize_attention(tokens, attention, layer=0, head=0, prefix="attention"):
    attn = attention[layer][0, head]
    token_list = tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn.cpu().numpy(),
        xticklabels=token_list,
        yticklabels=token_list,
        cmap='viridis',
        cbar=True,
    )
    plt.title(f'Attention Layer {layer}, Head {head}')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.tight_layout()
    plt.savefig(f'plots/{prefix}_layer{layer}_head{head}.png')
    plt.close()


if __name__ == "__main__":
    text = "The amazing movie won many awards"
    negative_text = "This movie was absolutely terrible and I hated it"
    tokens = tokenizer(text, return_tensors="pt")
    negative_tokens = tokenizer(negative_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**tokens)

    # Задача 2: формы
    print(f"Количество слоёв: {len(outputs.attentions)}")
    print(f"Форма для слоя 0: {outputs.attentions[0].shape}")

    # Эксперимент: сколько amazing смотрит на movie во всех 72 табличках
    for layer in range(6):
        for head in range(12):
            weight = outputs.attentions[layer][0, head][2, 3].item()
            print(f"совещание {layer}, группа {head}: {weight:.2f}")

    for head in range(12):
        visualize_attention(tokens, outputs.attentions, layer=0, head=head)

    with torch.no_grad():
        negative_outputs = model(**negative_tokens)

    visualize_attention(negative_tokens, negative_outputs.attentions, layer=5, head=0, prefix="negative")