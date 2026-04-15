from typing import Tuple

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from config import BASELINE_MODEL_NAME, LONGFORMER_MODEL_NAME


def build_baseline(num_labels: int = 2) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    构建 bert-base-chinese 基线模型及其 tokenizer。
    """
    model_name = BASELINE_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return model, tokenizer


def build_longformer(num_labels: int = 2) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """
    构建 Longformer 模型及其 tokenizer。

    默认使用 schen/longformer-chinese-base-4096。
    如果该 checkpoint 无法加载，可在 config.py 中替换 LONGFORMER_MODEL_NAME。
    """
    model_name = LONGFORMER_MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
    )
    return model, tokenizer

