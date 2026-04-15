from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import Dataset


class CommentDataset(Dataset):
    """
    从 CSV 文件读取 text 和 label 两列，并在 __getitem__ 中完成 tokenizer 编码。

    当 use_global_attention=True 时，会额外构造 global_attention_mask：
    - 第一个 token 位置设为 1
    - 其余位置设为 0

    适用于 Longformer 分类任务。
    """

    def __init__(
        self,
        csv_path: str,
        tokenizer,
        max_length: int = 256,
        use_global_attention: bool = False,
    ) -> None:
        self.df = pd.read_csv(csv_path)

        if "text" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError(
                f"数据集 {csv_path} 必须包含 'text' 和 'label' 两列，当前列为: {self.df.columns.tolist()}"
            )

        # 清理空值
        self.df = self.df.dropna(subset=["text", "label"]).reset_index(drop=True)

        self.texts: List[str] = self.df["text"].astype(str).tolist()
        self.labels: List[int] = self.df["label"].astype(int).tolist()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_global_attention = use_global_attention

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = int(self.labels[idx])

        # 保险检查：当前项目是二分类任务，标签必须是 0 或 1
        assert label in [0, 1], f"非法标签: {label}, idx={idx}"

        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        item: Dict[str, torch.Tensor] = {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long),
        }

        # 有些 tokenizer 会返回 token_type_ids，比如 BERT
        if "token_type_ids" in encoded:
            item["token_type_ids"] = encoded["token_type_ids"].squeeze(0)

        # Longformer 需要 global_attention_mask
        if self.use_global_attention:
            seq_len = item["input_ids"].shape[0]
            global_attention_mask = torch.zeros(seq_len, dtype=torch.long)
            global_attention_mask[0] = 1
            item["global_attention_mask"] = global_attention_mask

        return item