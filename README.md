<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg" alt="Status">

  <h1>Modeling Latent Financial Stress & Emotional Dynamics in Online Discourse</h1>
  <p>A Constraint-Optimized NLP Pipeline utilizing PEFT/LoRA, FinBERT, and BERTopic</p>
</div>

---

## 📖 Overview

The intersection of computational linguistics, behavioral finance, and psychology provides a profound mechanism to quantify systemic economic anxiety. This project introduces a fully functioning Natural Language Processing (NLP) pipeline designed to extract, model, and evaluate latent financial stress and emotional dynamics from online discourse.

Instead of relying on retrospective self-reporting (e.g., DASS-21), this architecture captures **real-time psychological metrics** from unstructured digital footprints across social platforms. By employing a **Parameter-Efficient Fine-Tuning (PEFT)** approach using Low-Rank Adaptation (LoRA) on domain-specific Transformers, the system operates efficiently within tight resource constraints without sacrificing state-of-the-art predictive performance.

---

## 🗄️ Datasets Used

This project dynamically pulls, normalizes, and merges four distinct, real-world Hugging Face datasets into an 11MB unified corpus.

1. **[twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment)**: Finance-specific short-form discourse (Bearish / Bullish / Neutral).
2. **[financial_phrasebank](https://huggingface.co/datasets/takala/financial_phrasebank)**: Expert-annotated financial news sentences (Negative / Neutral / Positive).
3. **[dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)**: 6-class social media emotion distributions from Twitter data.
4. **[go_emotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)**: 28-class fine-grained emotional taxonomy derived from Reddit.

**Preprocessing Strategy:**
- Preservation of capitalization, punctuation, and contextual syntax (for Transformer attention).
- Stripping of URLs, HTML artifacts, markdown formatting, and standardizing whitespaces.

---

## 🧠 System Architecture & Models

The architecture is divided into a robust sequential pipeline:

### 1. Baselines (Zero-Shot & Unsupervised)
* **VADER Sentiment**: Lexicon and rule-based sentiment analysis used to establish a zero-shot polarity baseline. Speed-optimized (inference at ~18ms per 500 texts).
* **Latent Dirichlet Allocation (LDA)**: Traditional bag-of-words topic modeling to discover broad semantic clusters (e.g., macro-finance vs. corporate activity).

### 2. Distilled Contextual Fine-Tuning (FinBERT with LoRA)
* **Base Model**: `ProsusAI/finbert` (110M parameters).
* **Optimization**: We implement Low-Rank Adaptation (LoRA) via Hugging Face's `peft` library, injecting trainable rank decomposition matrices into the `query` and `value` attention layers (`r=8, alpha=32`).
* **Resource Efficiency**: Reduces trainable parameters to **~297,219** (just 0.27% of the full model), ensuring total training time remains under 25 minutes on standard compute architectures while yielding immense performance gains over baselines.

### 3. Thematic Discovery (BERTopic)
* Employs `sentence-transformers/all-MiniLM-L6-v2` embeddings combined with UMAP dimensionality reduction and HDBSCAN clustering. This allows for dynamic, context-aware discovery of financial narratives evolving within the corpus.

---

## 📊 Performance & Evaluation Results

Final evaluations demonstrate the superiority of the PEFT approach over the classical lexicon approach.

| Model | Accuracy | F1-Macro | F1-Weighted | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: |
| **VADER (Baseline)** | 49.8% | 0.467 | 0.514 | 0.629 |
| **FinBERT-LoRA (Ours)** | **83.4%** | **0.786** | **0.835** | **0.937** |

*FinBERT-LoRA achieved an **86.6% improvement in F1-Macro** and a massive boost in categorical separability (ROC-AUC).*

### Generated Visualizations (`/outputs/`)
The integrated `visualize.py` metrics layer autonomously generates production-ready plots:
- `f1_comparison.png`: Side-by-side performance charting.
- `roc_curves_finbert.png` & `roc_curves_vader.png`: True Positive vs. False Positive tradeoff characteristics.
- `confusion_matrix_finbert.png` & `confusion_matrix_vader.png`: Granular class predictive stability.
- `topic_barchart.png`: Sizing and ranking of discovered financial narratives.

---

## 🚀 Installation and Execution

Clone the repository and install the dependencies to execute the fully autonomous pipeline.

```bash
# 1. Clone the repository
git clone https://github.com/abizer007/modeling-latent-financial-stress-online-discourse.git
cd modeling-latent-financial-stress-online-discourse

# 2. Setup Virtual Environment (Windows PowerShell shown)
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Execute Full Pipeline
python main.py
```

Execution will autonomously pull the datasets, normalize the corpus, execute the VADER baseline, pull the FinBERT architecture, inject the LoRA structures, train the model, inference against the test split, execute BERTopic, and save all graphs to `/outputs/`.

---

<div align="center">
<i>Built for resource-constrained, high-accuracy inference in financial cyberpsychology.</i>
</div>
