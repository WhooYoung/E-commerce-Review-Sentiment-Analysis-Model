import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import (
    DEVICE,
    LONGFORMER_BATCH_SIZE,
    LONGFORMER_EPOCHS,
    LONGFORMER_LR,
    LONGFORMER_MAX_LEN,
    BEST_LONGFORMER_MODEL_PATH,
    LONGFORMER_TRAIN_LOG,
    PROCESSED_DATA_DIR,
    ensure_dirs,
)
from dataset import CommentDataset
from models import build_longformer


def _move_batch_to_device(batch):
    return {k: v.to(DEVICE) for k, v in batch.items()}


def _remove_token_type_ids_if_needed(batch, model):
    """
    Longformer 模型通常不需要 token_type_ids，
    若 batch 中存在该键，则在前向传播前移除。
    """
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", "")
    if model_type and "longformer" in model_type and "token_type_ids" in batch:
        batch = dict(batch)
        batch.pop("token_type_ids", None)
    return batch


def train_one_epoch(model, dataloader, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Train-Long", ncols=80):
        batch = _move_batch_to_device(batch)
        batch = _remove_token_type_ids_if_needed(batch, model)
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


@torch.no_grad()
def evaluate_loss(model, dataloader) -> float:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Val-Long", ncols=80):
        batch = _move_batch_to_device(batch)
        batch = _remove_token_type_ids_if_needed(batch, model)
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

    # 如果 Longformer checkpoint 加载失败，应保留报错信息
    model, tokenizer = build_longformer(num_labels=2)
    model.to(DEVICE)

    train_dataset = CommentDataset(
        train_csv,
        tokenizer=tokenizer,
        max_length=LONGFORMER_MAX_LEN,
        use_global_attention=True,
    )
    val_dataset = CommentDataset(
        val_csv,
        tokenizer=tokenizer,
        max_length=LONGFORMER_MAX_LEN,
        use_global_attention=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=LONGFORMER_BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=LONGFORMER_BATCH_SIZE,
        shuffle=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LONGFORMER_LR)

    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(BEST_LONGFORMER_MODEL_PATH), exist_ok=True)

    log_fields = ["epoch", "train_loss", "val_loss"]
    log_file = LONGFORMER_TRAIN_LOG
    with open(log_file, "w", encoding="utf-8", newline="") as f_log:
        writer = csv.DictWriter(f_log, fieldnames=log_fields)
        writer.writeheader()

        for epoch in range(1, LONGFORMER_EPOCHS + 1):
            print(f"===== Longformer Epoch {epoch}/{LONGFORMER_EPOCHS} =====")
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

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": model.config._name_or_path,
                    },
                    BEST_LONGFORMER_MODEL_PATH,
                )
                print(f"保存最优 Longformer 模型到: {BEST_LONGFORMER_MODEL_PATH}")


if __name__ == "__main__":
    main()

