import os
from typing import Tuple

import gradio as gr
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
from predict import _remove_token_type_ids_if_needed


def _safe_load_model(
    model_type: str,
) -> Tuple[torch.nn.Module, object, int, str]:
    """
    Gradio 场景下，加载模型时要尽量给出友好提示。
    返回 (model, tokenizer, max_len, error_msg)，其中 error_msg 非空表示加载失败。
    """
    try:
        if model_type == "baseline":
            model, tokenizer = build_baseline(num_labels=2)
            max_len = BASELINE_MAX_LEN
            ckpt_path = BEST_BASELINE_MODEL_PATH
        else:
            model, tokenizer = build_longformer(num_labels=2)
            max_len = LONGFORMER_MAX_LEN
            ckpt_path = BEST_LONGFORMER_MODEL_PATH

        if not os.path.exists(ckpt_path):
            return (
                None,  # type: ignore
                None,  # type: ignore
                max_len,
                f"未找到 {model_type} 已训练模型权重，请先运行对应的训练脚本。",
            )

        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        model.to(DEVICE)
        model.eval()
        return model, tokenizer, max_len, ""
    except Exception as e:  # pragma: no cover - 主要做运行时友好提示
        return (
            None,  # type: ignore
            None,  # type: ignore
            0,
            f"加载 {model_type} 模型失败：{e}",
        )


class GradioApp:
    def __init__(self) -> None:
        self.baseline_model, self.baseline_tokenizer, self.baseline_max_len, self.baseline_err = _safe_load_model(
            "baseline"
        )
        self.long_model, self.long_tokenizer, self.long_max_len, self.long_err = _safe_load_model(
            "longformer"
        )

        self.id2label = {0: "negative", 1: "positive"}

    def predict(self, text: str, model_type: str):
        text = (text or "").strip()
        if not text:
            return "请输入文本。", 0.0, 0.0

        if model_type == "baseline":
            if self.baseline_err:
                return self.baseline_err, 0.0, 0.0
            model = self.baseline_model
            tokenizer = self.baseline_tokenizer
            max_len = self.baseline_max_len
            use_global_attention = False
        else:
            if self.long_err:
                return self.long_err, 0.0, 0.0
            model = self.long_model
            tokenizer = self.long_tokenizer
            max_len = self.long_max_len
            use_global_attention = True

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
        label_name = self.id2label.get(label_id, str(label_id))
        prob_neg, prob_pos = float(probs[0]), float(probs[1])
        return label_name, prob_neg, prob_pos


def main() -> None:
    app = GradioApp()

    with gr.Blocks() as demo:
        gr.Markdown("## 基于 Longformer 的电商长文本评论情感识别系统")
        with gr.Row():
            model_type = gr.Radio(
                choices=["baseline", "longformer"],
                value="baseline",
                label="选择模型",
            )
        text_input = gr.Textbox(
            lines=5,
            label="输入评论文本",
            placeholder="在此输入一条电商评论，例如：物流很快，包装很好，使用体验非常满意。",
        )
        btn = gr.Button("预测情感")

        label_output = gr.Textbox(label="预测标签（negative/positive）")
        prob_neg = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.001, label="prob_negative", interactive=False
        )
        prob_pos = gr.Slider(
            minimum=0.0, maximum=1.0, step=0.001, label="prob_positive", interactive=False
        )

        btn.click(
            fn=app.predict,
            inputs=[text_input, model_type],
            outputs=[label_output, prob_neg, prob_pos],
        )

    demo.launch()


if __name__ == "__main__":
    main()

