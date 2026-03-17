import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
_PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, labels, path):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(5, len(labels)), max(4, len(labels))))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    _save(fig, path)


def plot_roc_curves(y_true, y_prob, labels, path):
    classes   = list(range(len(labels)))
    y_bin     = label_binarize(y_true, classes=classes)
    fig, ax   = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(labels):
        if y_bin.ndim == 1:
            fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        else:
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_val:.2f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    _save(fig, path)


def plot_f1_bars(metrics_dict, path):
    models  = list(metrics_dict.keys())
    f1_vals = [metrics_dict[m].get("f1_macro", 0) for m in models]
    acc_vals = [metrics_dict[m].get("accuracy", 0) for m in models]

    x   = np.arange(len(models))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 2), 5))
    bars1 = ax.bar(x - w/2, f1_vals,  w, label="F1-Macro",  color=_PALETTE[0])
    bars2 = ax.bar(x + w/2, acc_vals, w, label="Accuracy",   color=_PALETTE[1])

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("F1-Macro & Accuracy Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    _save(fig, path)


def plot_vader_distribution(vader_df, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(vader_df["vader_compound"], bins=40, kde=True, color=_PALETTE[0], ax=axes[0])
    axes[0].axvline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title("VADER Compound Score Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Compound Score")
    axes[0].set_ylabel("Count")

    vader_df = vader_df.copy()
    vader_df["sentiment"] = vader_df["vader_compound"].apply(
        lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral")
    )
    counts = vader_df["sentiment"].value_counts()
    axes[1].pie(
        counts.values, labels=counts.index,
        autopct="%1.1f%%", colors=_PALETTE[:3],
        startangle=140, wedgeprops={"edgecolor": "white"},
    )
    axes[1].set_title("VADER Sentiment Breakdown", fontsize=13, fontweight="bold")
    _save(fig, path)


def plot_topic_barchart(topic_info, path, top_n=15):
    df = topic_info[topic_info["Topic"] != -1].head(top_n).copy()
    df["Name"] = df["Name"].str[:40]
    fig, ax = plt.subplots(figsize=(10, max(5, len(df) * 0.5)))
    bars = ax.barh(df["Name"][::-1], df["Count"][::-1], color=_PALETTE[0])
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 1, bar.get_y() + bar.get_height() / 2, f"{int(w)}", va="center", fontsize=9)
    ax.set_xlabel("Document Count", fontsize=12)
    ax.set_title(f"BERTopic — Top {top_n} Discovered Topics", fontsize=14, fontweight="bold")
    _save(fig, path)


def plot_per_class_f1(metrics_dict, label_names, path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(label_names))
    w = 0.35 / max(len(metrics_dict), 1)
    for idx, (model_name, m) in enumerate(metrics_dict.items()):
        per_cls = [m.get(f"f1_{l.lower()}", m.get(f"f1_class_{i}", 0)) for i, l in enumerate(label_names)]
        offset = (idx - len(metrics_dict) / 2) * w
        bars = ax.bar(x + offset, per_cls, w, label=model_name, color=_PALETTE[idx % len(_PALETTE)])
    ax.set_xticks(x)
    ax.set_xticklabels(label_names, rotation=20, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Score Comparison", fontsize=14, fontweight="bold")
    ax.legend()
    _save(fig, path)
