<div align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/Transformers-HuggingFace-yellow.svg" alt="Transformers">
  <img src="https://img.shields.io/badge/Status-Complete-success.svg" alt="Status">

  <h1>Modeling Latent Financial Stress & Emotional Dynamics in Online Discourse</h1>
  <p>A Constraint-Optimized NLP Pipeline utilizing PEFT/LoRA, FinBERT, Roberta, and BERTopic</p>
</div>

---

## 📖 Overview

The intersection of computational linguistics, behavioral finance, and psychology provides a profound mechanism to quantify systemic economic anxiety. This project introduces a fully functioning Natural Language Processing (NLP) pipeline designed to extract, model, and evaluate latent financial stress and emotional dynamics from online discourse.

Instead of relying on retrospective self-reporting (e.g., DASS-21), this architecture captures **real-time psychological metrics** from unstructured digital footprints across social platforms. By employing a **Parameter-Efficient Fine-Tuning (PEFT)** approach using Low-Rank Adaptation (LoRA) on domain-specific and robust Transformers (FinBERT & Roberta), the system operates efficiently within tight resource constraints without sacrificing state-of-the-art predictive performance.

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

### 2. Distilled Contextual Fine-Tuning (FinBERT & Roberta with LoRA)
* **Base Models**: `ProsusAI/finbert` (110M parameters) and `roberta-base` (125M parameters).
* **Optimization**: We implement Low-Rank Adaptation (LoRA) via Hugging Face's `peft` library, injecting trainable rank decomposition matrices into the `query` and `value` attention layers (`r=8, alpha=32`).
* **Resource Efficiency**: Reduces trainable parameters to **~297,219** (just 0.2% of the full models), ensuring rapid training times on standard compute architectures while yielding immense performance gains over baselines.

### 3. Thematic Discovery (BERTopic)
* Employs `sentence-transformers/all-MiniLM-L6-v2` embeddings combined with UMAP dimensionality reduction and HDBSCAN clustering. This allows for dynamic, context-aware discovery of financial narratives evolving within the corpus.

---

## 📊 Performance & Evaluation Results

Final evaluations demonstrate the superiority of the PEFT approach over the classical lexicon approach, with **Roberta-LoRA** emerging as the optimal architecture for resolving nuanced financial stress in online discourse.

| Model | Accuracy | F1-Macro | F1-Weighted | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: |
| **VADER (Baseline)** | 49.8% | 0.467 | 0.514 | 0.629 |
| **FinBERT-LoRA** | 83.1% | 0.783 | 0.831 | 0.938 |
| **Roberta-LoRA (Best)** | **88.6%** | **0.859** | **0.887** | **0.961** |

*Roberta-LoRA achieved an **nearly 86% F1-Macro score** and a massive boost in categorical separability (ROC-AUC 0.961). Its performance outstripped FinBERT particularly in determining volatile "Bearish" contexts (+11.9% absolute improvement in F1).*

### 📈 System Metrics & Visualizations

The integrated metrics layer autonomously generates production-ready comparative plots.

#### 📊 Model Performance Comparison
<div align="center">
  <img src="outputs/f1_comparison.png" alt="F1 Comparison" width="45%">
  <img src="outputs/per_class_f1.png" alt="Per-Class F1 Score Comparison" width="45%">
</div>
<br/>
<div align="center">
  <em>Figure 1: (Left) Macro and Weighted F1-Scores across baselines and fine-tuned models. (Right) Class-wise F1 precision mappings explicitly comparing Model performance in bullish, bearish, and neutral contexts.</em>
</div>

#### 🎯 Predictive Reliability (Confusion Matrices)
<div align="center">
  <img src="outputs/confusion_matrix_roberta.png" alt="Roberta Confusion Matrix" width="32%">
  <img src="outputs/confusion_matrix_finbert.png" alt="FinBERT Confusion Matrix" width="32%">
  <img src="outputs/confusion_matrix_vader.png" alt="VADER Confusion Matrix" width="32%">
</div>
<br/>
<div align="center">
  <em>Figure 2: Granular predictive stability. Roberta-LoRA (left) excels in isolating bearish contexts over FinBERT (center), both significantly outperforming the VADER zero-shot baseline (right).</em>
</div>

#### 📉 Receiver Operating Characteristics (ROC)
<div align="center">
  <img src="outputs/roc_curves_roberta.png" alt="Roberta ROC Curve" width="32%">
  <img src="outputs/roc_curves_finbert.png" alt="FinBERT ROC Curve" width="32%">
  <img src="outputs/roc_curves_vader.png" alt="VADER ROC Curve" width="32%">
</div>
<br/>
<div align="center">
  <em>Figure 3: True Positive versus False Positive tradeoff characteristics. Demonstrates the categorical separability of the Roberta models (Left) vs baseline models.</em>
</div>

#### 🧠 Semantic Topic Modeling
<div align="center">
  <img src="outputs/topic_barchart.png" alt="Identified Financial Narratives" width="60%">
</div>
<br/>
<div align="center">
  <em>Figure 4: Sizing and ranking of contextual financial narratives natively discovered within the corpus by BERTopic.</em>
</div>

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

Execution will autonomously pull the datasets, normalize the corpus, execute the VADER baseline, pull the Transformer architectures (FinBERT and Roberta), inject the LoRA structures, train both models sequentially, run inference against the test split, execute BERTopic, and save all comparative graphs safely to `/outputs/`.

---

<div align="center">
<i>Built for resource-constrained, high-accuracy inference in financial cyberpsychology.</i>
</div>
