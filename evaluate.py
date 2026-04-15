import argparse
import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import (
    BASELINE_MAX_LEN,
    BEST_BASELINE_MODEL_PATH,
    BEST_LONGFORMER_MODEL_PATH,
    DEVICE,
    LONGFORMER_MAX_LEN,
    PROCESSED_DATA_DIR,
)
from dataset import CommentDataset
from models import build_baseline, build_longformer


def _load_model_and_tokenizer(
    model_type: str,
) -> Tuple[torch.nn.Module, object, int]:
    if model_type == "baseline":
        model, tokenizer = build_baseline(num_labels=2)
        max_len = BASELINE_MAX_LEN
        ckpt_path = BEST_BASELINE_MODEL_PATH
    elif model_type == "longformer":
        model, tokenizer = build_longformer(num_labels=2)
        max_len = LONGFORMER_MAX_LEN
        ckpt_path = BEST_LONGFORMER_MODEL_PATH
    else:
        raise ValueError("model_type 必须是 'baseline' 或 'longformer'")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"未找到已训练好的 {model_type} 模型权重，请先运行相应的训练脚本。\n路径: {ckpt_path}"
        )

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, max_len


def _remove_token_type_ids_if_needed(batch, model):
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "")
    if model_type and "longformer" in model_type and "token_type_ids" in batch:
        batch = dict(batch)
        batch.pop("token_type_ids", None)
    return batch


@torch.no_grad()
def evaluate_model(model, dataloader) -> Tuple[np.ndarray, np.ndarray]:
    all_preds = []
    all_labels = []
    for batch in dataloader:
        labels = batch["labels"].numpy()
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        batch = _remove_token_type_ids_if_needed(batch, model)
        outputs = model(**batch)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.append(preds)
        all_labels.append(labels)
    return np.concatenate(all_labels), np.concatenate(all_preds)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["baseline", "longformer"],
        help="选择评估的模型类型：baseline 或 longformer",
    )
    args = parser.parse_args()

    test_csv = os.path.join(PROCESSED_DATA_DIR, "test.csv")
    if not os.path.exists(test_csv):
        raise FileNotFoundError(
            f"找不到测试集文件 {test_csv}，请先运行 prepare_data.py。"
        )

    model, tokenizer, max_len = _load_model_and_tokenizer(args.model_type)

    use_global_attention = args.model_type == "longformer"
    dataset = CommentDataset(
        test_csv,
        tokenizer=tokenizer,
        max_length=max_len,
        use_global_attention=use_global_attention,
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    y_true, y_pred = evaluate_model(model, dataloader)

    print("===== classification_report =====")
    print(classification_report(y_true, y_pred, digits=4))

    print("===== confusion_matrix =====")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()

