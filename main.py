import os
import sys
import logging
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data.dataset_manager import DatasetManager
from src.models.baselines import run_vader_baseline, run_lda_baseline, display_lda_topics
from src.models.lora_finetune import FinBERTTrainer
from src.models.roberta_finetune import RobertaTrainer
from src.models.topic_modeling import run_bertopic_clustering
from src.evaluation.metrics import (
    compute_all_metrics,
    evaluate_inference_speed,
    generate_comparative_report,
)
from src.evaluation.visualize import (
    plot_confusion_matrix,
    plot_roc_curves,
    plot_f1_bars,
    plot_vader_distribution,
    plot_topic_barchart,
    plot_per_class_f1,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

OUTPUTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUTPUTS, exist_ok=True)

FINANCIAL_SOURCES = ["twitter_fin_sentiment", "financial_phrasebank"]
VADER_LABELS      = ["Negative", "Neutral", "Positive"]
VADER_LABEL2ID    = {"Negative": 0, "Neutral": 1, "Positive": 2}


def _vader_bucket(compound):
    if compound > 0.05:
        return 2
    if compound < -0.05:
        return 0
    return 1


def _sentiment_label_to_vader_id(lbl):
    mapping = {
        "Bearish": 0, "Negative": 0,
        "Neutral": 1,
        "Bullish": 2, "Positive": 2,
    }
    return mapping.get(lbl, 1)


def _norm_probs(arr):
    s = arr.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return arr / s


def run_pipeline():
    logger.info("=" * 65)
    logger.info("  Financial Stress NLP Pipeline — Full Execution")
    logger.info("=" * 65)

    dm = DatasetManager(use_cached=True, max_tokens=128)
    dm.build()

    if dm.corpus.empty:
        logger.error("Corpus empty — aborting.")
        return

    logger.info(f"\nCorpus summary:\n{dm.summary().to_string(index=False)}\n")
    train_df, val_df, test_df = dm.splits()

    fin_train = train_df[train_df["source"].isin(FINANCIAL_SOURCES)].reset_index(drop=True)
    fin_val   = val_df[val_df["source"].isin(FINANCIAL_SOURCES)].reset_index(drop=True)
    fin_test  = test_df[test_df["source"].isin(FINANCIAL_SOURCES)].reset_index(drop=True)

    if len(fin_train) < 100:
        logger.warning("Low financial subset — using full corpus with 3-class remap.")
        fin_train = train_df.sample(min(2000, len(train_df)), random_state=42)
        fin_val   = val_df.sample(min(400, len(val_df)), random_state=42)
        fin_test  = test_df.sample(min(800, len(test_df)), random_state=42)

    texts_full  = dm.texts
    texts_train = train_df["text_clean"].tolist()

    fin_test_texts  = fin_test["text_clean"].tolist()
    fin_y_true_vader = [_sentiment_label_to_vader_id(lbl) for lbl in fin_test["label_text"].tolist()]

    logger.info("\n── PHASE 2a: VADER Baseline ─────────────────────────────────")
    vader_results, vader_time = evaluate_inference_speed(run_vader_baseline, fin_test_texts)
    vader_preds = vader_results["vader_compound"].apply(_vader_bucket).tolist()
    vader_probs = _norm_probs(np.column_stack([
        vader_results["vader_neg"],
        vader_results["vader_neu"],
        vader_results["vader_pos"],
    ]))
    vader_metrics = compute_all_metrics(fin_y_true_vader, vader_preds, vader_probs, VADER_LABELS)
    vader_metrics["inference_time_s"] = round(vader_time, 4)
    logger.info(f"VADER inference: {vader_time:.4f}s | F1-macro: {vader_metrics['f1_macro']:.4f}")
    plot_vader_distribution(vader_results, os.path.join(OUTPUTS, "vader_distribution.png"))
    logger.info("Saved: vader_distribution.png")

    logger.info("\n── PHASE 2b: LDA Baseline ───────────────────────────────────")
    lda_model, tf_vectorizer = run_lda_baseline(texts_train[:3000], n_topics=7)
    if lda_model:
        display_lda_topics(lda_model, tf_vectorizer.get_feature_names_out(), no_top_words=8)

    logger.info("\n── PHASE 3: FinBERT LoRA Fine-Tuning ────────────────────────")
    finbert = FinBERTTrainer(num_labels=3, lora_r=8, lora_alpha=32, max_length=128)
    finbert.train(fin_train, fin_val, output_dir=os.path.join(OUTPUTS, "finbert_lora"), epochs=3)

    logger.info("\n── PHASE 3: FinBERT Inference on Financial Test Set ─────────")
    fin_preds, fin_probs, fin_pred_labels = finbert.predict(fin_test_texts)
    fin_label_names = list(finbert.label2id.keys())
    fin_y_true_mapped = [finbert.label2id.get(lbl, 0) for lbl in fin_test["label_text"].tolist()]
    finbert_metrics = compute_all_metrics(fin_y_true_mapped, fin_preds, fin_probs, fin_label_names)
    finbert_metrics["inference_time_s"] = "N/A"

    logger.info("\n── PHASE 3b: Roberta LoRA Fine-Tuning ───────────────────────")
    roberta = RobertaTrainer(num_labels=3, lora_r=8, lora_alpha=32, max_length=128)
    roberta.train(fin_train, fin_val, output_dir=os.path.join(OUTPUTS, "roberta_lora"), epochs=3)

    logger.info("\n── PHASE 3b: Roberta Inference on Financial Test Set ────────")
    rob_preds, rob_probs, rob_pred_labels = roberta.predict(fin_test_texts)
    rob_label_names = list(roberta.label2id.keys())
    rob_y_true_mapped = [roberta.label2id.get(lbl, 0) for lbl in fin_test["label_text"].tolist()]
    roberta_metrics = compute_all_metrics(rob_y_true_mapped, rob_preds, rob_probs, rob_label_names)
    roberta_metrics["inference_time_s"] = "N/A"

    logger.info("\n── PHASE 4: BERTopic ────────────────────────────────────────")
    topic_model, topic_info = run_bertopic_clustering(texts_full[:3000])
    if topic_info is not None:
        logger.info(f"Top Topics:\n{topic_info.head(7).to_string()}")
        plot_topic_barchart(topic_info, os.path.join(OUTPUTS, "topic_barchart.png"), top_n=15)
        logger.info("Saved: topic_barchart.png")

    logger.info("\n── PHASE 5: Evaluation & Visualization ──────────────────────")
    all_model_metrics = {
        "VADER (baseline)": vader_metrics,
        "FinBERT-LoRA":     finbert_metrics,
        "Roberta-LoRA":     roberta_metrics,
    }
    report_df = generate_comparative_report(all_model_metrics)
    report_df.to_csv(os.path.join(OUTPUTS, "metrics_summary.csv"), index=False)
    logger.info("Saved: metrics_summary.csv")

    plot_f1_bars(all_model_metrics, os.path.join(OUTPUTS, "f1_comparison.png"))
    logger.info("Saved: f1_comparison.png")

    shared_labels = fin_label_names if len(fin_label_names) <= len(VADER_LABELS) else VADER_LABELS
    plot_per_class_f1(all_model_metrics, shared_labels, os.path.join(OUTPUTS, "per_class_f1.png"))
    logger.info("Saved: per_class_f1.png")

    plot_confusion_matrix(fin_y_true_mapped, fin_preds, fin_label_names, os.path.join(OUTPUTS, "confusion_matrix_finbert.png"))
    logger.info("Saved: confusion_matrix_finbert.png")

    plot_confusion_matrix(rob_y_true_mapped, rob_preds, rob_label_names, os.path.join(OUTPUTS, "confusion_matrix_roberta.png"))
    logger.info("Saved: confusion_matrix_roberta.png")

    plot_confusion_matrix(fin_y_true_vader, vader_preds, VADER_LABELS, os.path.join(OUTPUTS, "confusion_matrix_vader.png"))
    logger.info("Saved: confusion_matrix_vader.png")

    if fin_probs is not None and fin_probs.shape[1] == len(fin_label_names):
        plot_roc_curves(fin_y_true_mapped, fin_probs, fin_label_names, os.path.join(OUTPUTS, "roc_curves_finbert.png"))
        logger.info("Saved: roc_curves_finbert.png")

    if rob_probs is not None and rob_probs.shape[1] == len(rob_label_names):
        plot_roc_curves(rob_y_true_mapped, rob_probs, rob_label_names, os.path.join(OUTPUTS, "roc_curves_roberta.png"))
        logger.info("Saved: roc_curves_roberta.png")

    if vader_probs is not None:
        plot_roc_curves(fin_y_true_vader, vader_probs, VADER_LABELS, os.path.join(OUTPUTS, "roc_curves_vader.png"))
        logger.info("Saved: roc_curves_vader.png")

    logger.info(f"\n{'='*65}")
    logger.info(f"  Pipeline complete. Outputs saved to /outputs/")
    for f in sorted(os.listdir(OUTPUTS)):
        if f.endswith(".png") or f.endswith(".csv"):
            logger.info(f"    {f}")
    logger.info(f"{'='*65}\n")


if __name__ == "__main__":
    run_pipeline()
