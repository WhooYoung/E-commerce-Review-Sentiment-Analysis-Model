import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    BASELINE_MAX_LEN,
    BEST_BASELINE_MODEL_PATH,
    BEST_LONGFORMER_MODEL_PATH,
    DEVICE,
    EXCEL_RESULT_DIR,
    LONGFORMER_MAX_LEN,
    ensure_dirs,
)
from models import build_baseline, build_longformer
from predict import _remove_token_type_ids_if_needed


def _load_model_and_tokenizer(model_type: str):
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
            f"未找到 {model_type} 模型权重，请先运行对应的训练脚本。\n路径: {ckpt_path}"
        )

    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, tokenizer, max_len


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="输入 Excel 文件路径")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="输出 Excel 文件路径（可为空，为空则自动生成到 outputs/excel_results/）",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["baseline", "longformer"],
    )
    args = parser.parse_args()

    ensure_dirs()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"输入文件不存在: {args.input}")

    df = pd.read_excel(args.input)
    if "text" not in df.columns:
        raise ValueError("输入 Excel 必须包含 'text' 列。")

    model, tokenizer, max_len = _load_model_and_tokenizer(args.model_type)
    use_global_attention = args.model_type == "longformer"

    id2label = {0: "negative", 1: "positive"}

    pred_labels = []
    prob_negatives = []
    prob_positives = []

    texts = df["text"].astype(str).tolist()

    for text in tqdm(texts, desc="Batch Inference", ncols=80):
        encoded = tokenizer(
            text,
            max_length=max_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        inputs = {
            "input_ids": encoded["input_ids"].to(DEVICE),
            "attention_mask": encoded["attention_mask"].to(DEVICE),
        }
        if "token_type_ids" in encoded:
            inputs["token_type_ids"] = encoded["token_type_ids"].to(DEVICE)

        if use_global_attention:
            seq_len = inputs["input_ids"].shape[1]
            global_attention_mask = torch.zeros_like(inputs["attention_mask"])
            global_attention_mask[:, 0] = 1
            inputs["global_attention_mask"] = global_attention_mask

        inputs = _remove_token_type_ids_if_needed(inputs, model)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        label_id = int(np.argmax(probs))
        label_name = id2label.get(label_id, str(label_id))

        pred_labels.append(label_name)
        prob_negatives.append(float(probs[0]))
        prob_positives.append(float(probs[1]))

    df["pred_label"] = pred_labels
    df["prob_negative"] = prob_negatives
    df["prob_positive"] = prob_positives

    if args.output:
        output_path = args.output
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(EXCEL_RESULT_DIR, f"batch_result_{ts}.xlsx")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_excel(output_path, index=False)
    print(f"批量预测结果已保存到: {output_path}")


if __name__ == "__main__":
    main()

