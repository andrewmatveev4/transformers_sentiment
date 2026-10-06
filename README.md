# Sentiment Analysis with DistilBERT: frozen embeddings vs fine-tuning

A 7-day hands-on project exploring how transformers work, from tokenization to a deployed demo.
The core experiment compares two ways of using `distilbert-base-uncased` for binary sentiment
classification on SST-2:

- **Baseline:** frozen DistilBERT, `[CLS]` embeddings + logistic regression
- **Fine-tuned:** the whole model trained end-to-end with a classification head

**Result:** fine-tuning cut the number of errors by **38%** (286 → 178 on 2,000 test reviews).

## Results

| Model | Macro F1 | Accuracy | Errors (of 2,000) |
|---|---|---|---|
| Baseline: frozen DistilBERT + LogReg | 0.854 | 0.857 | 286 |
| Fine-tuned DistilBERT (3 epochs) | **0.909** | **0.911** | **178** |

- Relative F1 improvement: +6.5%
- 108 fewer errors, both error types reduced (negative→positive: 156 → 101, positive→negative: 130 → 77)
- All models were evaluated on the **same** 2,000 reviews. Verified by reproducing the
  Day 4 and Day 5 scores exactly in the comparison script

Confusion matrices: `plots/cm_baseline.png`, `plots/cm_finetuned.png`

## Project structure

| File | Day | What it does |
|---|---|---|
| `day01_tokenization.py` | 1 | Tokenizer, batching with padding and attention mask, special tokens |
| `day02_embeddings.py` | 2 | Hidden states, `[CLS]` embeddings, cosine similarity |
| `day03_attention.py` | 3 | Attention heatmaps across layers and heads (`plots/attention_*.png`) |
| `prepare_data.py` | 4 | Downloads SST-2, samples 10,000 reviews |
| `day04_baseline.py` | 4 | Logistic regression on frozen `[CLS]` embeddings |
| `day05_finetune.py` | 5 | Fine-tuning loop (PyTorch, AdamW, Apple MPS) |
| `day06_compare.py` | 6 | Inference functions, metrics, confusion matrices |
| `day07_errors.py` | 7 | Error analysis, most confident mistakes |
| `app.py` | 7 | Gradio demo |

## Key findings

1. **Frozen embeddings barely separate sentiment by cosine similarity.** "Great movie!" vs
   "Terrible film!" scored 0.984, almost as high as two positive phrases (0.995). DistilBERT was
   pretrained to predict masked words, and in that task *great* and *terrible* are interchangeable.
   Still, logistic regression reached 0.854 F1: it learns which few of the 768 dimensions carry sentiment.
2. **Attention maps show mechanics, not explanations.** Most heads in early layers either attend
   to the next token or dump attention onto `[CLS]` / `[SEP]`. A weight of 1.00 for
   *amazing → movie* turned out to be a "look at the next word" head, not a semantic link.
3. **Fine-tuning gave +5.5 F1 points** in about 4 minutes of training on a MacBook (Apple MPS).
   Without a fixed seed, run-to-run variance was about 0.005 F1, so smaller differences are noise.
4. **Both models fail the same way:** they mistake negative reviews for positive more often than
   the reverse.

## Error analysis (fine-tuned model)

Full report: `error_analysis.txt`

- **Sarcasm and irony:** praise words in mocking phrases are read as praise (*"worthy of the gong"*)
- **Negation in idioms:** *"Can't recommend it enough!"* is classified as **97.8% negative**.
  The same result with SST-2-style formatting (*"ca n't recommend it enough ."*), so the issue is
  the idiom itself, not the input format
- **Surface words instead of meaning:** *"laughable"* is predicted positive because of *"laugh"*
- **Rare words and typos** get split into meaningless subword pieces (*"revigorates"*, *"venality"*)
- **Not every error is the model's fault:** SST-2 consists of phrase fragments without context
  (*"of empathy"*, *"tossed in"*), and some labels are debatable
- **Errors are confident:** most top errors have probability 0.999 or 0.002, so a confidence
  threshold cannot catch them. The model is poorly calibrated
- **Text length does not matter:** 53 characters for errors vs 54 on average

## Demo

```bash
python app.py
```

Open http://127.0.0.1:7860

The demo shows the predicted class and both probabilities. If confidence is below 0.7, it answers
"uncertain" instead of forcing a label. This works for texts with no sentiment signal
(e.g. *"ulala"* → 61% positive → uncertain), but not for the confident errors listed above.

## How to reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python prepare_data.py       # SST-2 sample → data/
python day04_baseline.py     # baseline, saves baseline_model.pkl
python day05_finetune.py     # fine-tuning, saves fine_tuned_model/ (~270 MB, not in the repo)
python day06_compare.py      # comparison, confusion matrices
python day07_errors.py       # error analysis
python app.py                # demo
```

## Deviations from the course assignment

- **10,000-review SST-2 sample** instead of the full 67k, for speed
- **`max_length=64`** for fine-tuning instead of 128: SST-2 phrases are short, so 128 mostly adds padding
- **Apple MPS** instead of CUDA
- **`torch.optim.AdamW`**: `transformers.AdamW` was removed in recent versions
- **12 attention heads**, not 8 as stated in the assignment (confirmed by tensor shape `[1, 12, seq, seq]`)
- **Baseline is frozen DistilBERT + LogReg**, not TF-IDF, so `predict_baseline` uses `[CLS]` embeddings instead of a vectorizer
- **`label_map` fixed to two classes.** The assignment's three-class map would display every positive review as "Neutral"
- **`n_jobs` removed** from `LogisticRegression` (no effect since scikit-learn 1.8)

## Limitations

- Binary classification only: no neutral class
- Poor calibration: the model is equally confident when right and when wrong
- SST-2 text is pre-tokenized and lowercased, which differs from how real users write
- Trained on a 10k subset with a single run

## Tech stack

Python 3.14, PyTorch, Hugging Face Transformers and Datasets, scikit-learn, Gradio, matplotlib, seaborn
