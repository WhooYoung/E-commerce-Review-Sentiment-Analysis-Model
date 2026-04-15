import argparse
import os
from typing import Dict, Tuple

import numpy as np
import torch

from config import (
    BASELINE_MAX_LEN,
    BEST_BASELINE_MODEL_PATH,
    BEST_LONGFORMER_MODEL_PATH,
    DEVICE,
    LONGFORMER_MAX_LEN,
)
from models import build_baseline, build_longformer


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


def _remove_token_type_ids_if_needed(inputs: Dict[str, torch.Tensor], model):
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "")
    if model_type and "longformer" in model_type and "token_type_ids" in inputs:
        inputs = dict(inputs)
        inputs.pop("token_type_ids", None)
    return inputs


@torch.no_grad()
def predict_single(
    text: str,
    model,
    tokenizer,
    max_length: int,
    use_global_attention: bool,
) -> Tuple[int, str, np.ndarray]:
    encoded = tokenizer(
        text,
        max_length=max_length,
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

    outputs = model(**inputs)
    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
    label_id = int(np.argmax(probs))

    # 这里假设 0 为负向，1 为正向
    id2label = {0: "negative", 1: "positive"}
    label_name = id2label.get(label_id, str(label_id))
    return label_id, label_name, probs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=["baseline", "longformer"],
    )
    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="待预测的中文评论文本",
    )
    args = parser.parse_args()

    model, tokenizer, max_len = _load_model_and_tokenizer(args.model_type)
    use_global_attention = args.model_type == "longformer"

    label_id, label_name, probs = predict_single(
        args.text,
        model,
        tokenizer,
        max_length=max_len,
        use_global_attention=use_global_attention,
    )

    print("===== 单条预测结果 =====")
    print(f"text: {args.text}")
    print(f"label_id: {label_id}")
    print(f"label_name: {label_name}")
    print(f"probabilities (neg, pos): {probs}")


if __name__ == "__main__":
    main()

