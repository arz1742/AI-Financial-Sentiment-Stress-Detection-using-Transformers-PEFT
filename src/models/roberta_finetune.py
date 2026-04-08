import os
import torch
import logging
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

_OUTPUTS = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "outputs"))
os.makedirs(_OUTPUTS, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class RobertaTrainer:
    MODEL_NAME = "roberta-base"

    def __init__(self, num_labels=3, lora_r=8, lora_alpha=32, max_length=128):
        self.num_labels  = num_labels
        self.max_length  = max_length
        self.lora_r      = lora_r
        self.lora_alpha  = lora_alpha
        self.tokenizer   = None
        self.model       = None
        self.trainer     = None
        self.label2id    = {}
        self.id2label    = {}

    def _build_model(self):
        logger.info(f"Loading {self.MODEL_NAME} and wrapping with LoRA (r={self.lora_r}, alpha={self.lora_alpha}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        base_model = AutoModelForSequenceClassification.from_pretrained(
            self.MODEL_NAME,
            num_labels=self.num_labels,
            ignore_mismatched_sizes=True,
        )
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            lora_dropout=0.1,
            target_modules=["query", "value"],
        )
        self.model = get_peft_model(base_model, peft_config)
        self.model.print_trainable_parameters()

    def _tokenize(self, batch):
        return self.tokenizer(
            batch["text"],
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

    def _compute_metrics(self, eval_pred):
        f1_metric  = evaluate.load("f1")
        acc_metric = evaluate.load("accuracy")
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {
            "f1_macro":   f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
            "f1_weighted": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
            "accuracy":   acc_metric.compute(predictions=preds, references=labels)["accuracy"],
        }

    def train(self, train_df, val_df, output_dir=None, epochs=3):
        if output_dir is None:
            output_dir = os.path.join(_OUTPUTS, "roberta_lora")
        self._build_model()

        label_names = sorted(train_df["label_text"].unique())
        self.label2id = {l: i for i, l in enumerate(label_names)}
        self.id2label = {i: l for l, i in self.label2id.items()}

        def make_hf(df):
            sub = df[["text_clean", "label_text"]].rename(columns={"text_clean": "text"})
            sub = sub[sub["text"].str.strip().str.len() > 0].copy()
            sub["label"] = sub["label_text"].map(self.label2id)
            sub = sub.dropna(subset=["label"])
            sub["label"] = sub["label"].astype(int)
            return Dataset.from_pandas(sub[["text", "label"]].reset_index(drop=True))

        train_ds = make_hf(train_df).map(self._tokenize, batched=True)
        val_ds   = make_hf(val_df).map(self._tokenize, batched=True)

        use_fp16 = torch.cuda.is_available()
        args = TrainingArguments(
            output_dir                  = output_dir,
            num_train_epochs            = epochs,
            per_device_train_batch_size = 16,
            per_device_eval_batch_size  = 32,
            gradient_accumulation_steps = 2,
            eval_strategy               = "epoch",
            save_strategy               = "epoch",
            logging_steps               = 50,
            fp16                        = use_fp16,
            learning_rate               = 2e-4,
            weight_decay                = 0.01,
            load_best_model_at_end      = True,
            metric_for_best_model       = "f1_macro",
            report_to                   = "none",
        )
        self.trainer = Trainer(
            model             = self.model,
            args              = args,
            train_dataset     = train_ds,
            eval_dataset      = val_ds,
            processing_class  = self.tokenizer,
            data_collator     = DataCollatorWithPadding(self.tokenizer),
            compute_metrics   = self._compute_metrics,
        )
        logger.info("Starting Roberta LoRA training ...")
        self.trainer.train()
        logger.info("Training complete.")
        return self.trainer

    def predict(self, texts):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not trained. Call .train() first.")
        self.model.eval()
        device = get_device()
        self.model.to(device)
        all_logits = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                truncation=True,
                max_length=self.max_length,
                padding=True,
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                out = self.model(**enc)
            all_logits.append(out.logits.cpu().numpy())
        logits     = np.vstack(all_logits)
        probs      = torch.softmax(torch.tensor(logits), dim=-1).numpy()
        preds      = np.argmax(logits, axis=-1)
        pred_labels = [self.id2label[p] for p in preds]
        return preds, probs, pred_labels
