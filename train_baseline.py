import csv
import os
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    BASELINE_BATCH_SIZE,
    BASELINE_EPOCHS,
    BASELINE_LR,
    BASELINE_MAX_LEN,
    BEST_BASELINE_MODEL_PATH,
    DEVICE,
    PROCESSED_DATA_DIR,
    BASELINE_TRAIN_LOG,
    ensure_dirs,
)
from dataset import CommentDataset
from models import build_baseline


def train_one_epoch(
    model,
    dataloader: DataLoader,
    optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train", ncols=80):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_loss(model, dataloader: DataLoader) -> float:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Val  ", ncols=80):
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def main() -> None:
    ensure_dirs()

    train_csv = os.path.join(PROCESSED_DATA_DIR, "train.csv")
    val_csv = os.path.join(PROCESSED_DATA_DIR, "val.csv")

    if not os.path.exists(train_csv) or not os.path.exists(val_csv):
        raise FileNotFoundError(
            f"找不到预处理数据文件，请先运行 prepare_data.py。\n"
            f"期望位置: {train_csv}, {val_csv}"
        )

    model, tokenizer = build_baseline(num_labels=2)
    model.to(DEVICE)

    train_dataset = CommentDataset(
        train_csv,
        tokenizer=tokenizer,
        max_length=BASELINE_MAX_LEN,
        use_global_attention=False,
    )
    val_dataset = CommentDataset(
        val_csv,
        tokenizer=tokenizer,
        max_length=BASELINE_MAX_LEN,
        use_global_attention=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BASELINE_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BASELINE_BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASELINE_LR)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(BEST_BASELINE_MODEL_PATH), exist_ok=True)

    # 训练日志
    log_fields = ["epoch", "train_loss", "val_loss"]
    log_file = BASELINE_TRAIN_LOG
    with open(log_file, "w", encoding="utf-8", newline="") as f_log:
        writer = csv.DictWriter(f_log, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, BASELINE_EPOCHS + 1):
            print(f"===== Epoch {epoch}/{BASELINE_EPOCHS} =====")
            train_loss = train_one_epoch(model, train_loader, optimizer)
            val_loss = evaluate_loss(model, val_loader)

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                }
            )
            f_log.flush()

            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model.config._name_or_path,
                    },
                    BEST_BASELINE_MODEL_PATH,
                )
                print(f"保存最优模型到: {BEST_BASELINE_MODEL_PATH}")


if __name__ == "__main__":
    main()

