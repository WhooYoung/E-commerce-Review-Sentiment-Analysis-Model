from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os

SAVE_DIR = r"E:\ecommerce_longformer\data\processed"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(SAVE_DIR, "train.csv")
VAL_PATH = os.path.join(SAVE_DIR, "val.csv")
TEST_PATH = os.path.join(SAVE_DIR, "test.csv")


def main():
    print("开始加载 ndiy/ChnSentiCorp 数据集...")
    ds = load_dataset("ndiy/ChnSentiCorp")

    print("数据集分片：", ds)

    train_df = pd.DataFrame(ds["train"])
    test_df = pd.DataFrame(ds["test"])

    print("train 列名：", train_df.columns.tolist())
    print(train_df.head())

    if "text" not in train_df.columns or "label" not in train_df.columns:
        raise ValueError(f"当前数据列不符合预期：{train_df.columns.tolist()}")

    train_df = train_df[["text", "label"]].copy()
    test_df = test_df[["text", "label"]].copy()

    train_df = train_df.dropna(subset=["text", "label"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=["text", "label"]).reset_index(drop=True)

    train_df["text"] = train_df["text"].astype(str)
    test_df["text"] = test_df["text"].astype(str)
    train_df["label"] = train_df["label"].astype(int)
    test_df["label"] = test_df["label"].astype(int)

    # 从 train 再切一部分出来做 val
    train_df, val_df = train_test_split(
        train_df,
        test_size=0.1,
        random_state=42,
        stratify=train_df["label"]
    )

    print("train label 分布：")
    print(train_df["label"].value_counts())
    print("val label 分布：")
    print(val_df["label"].value_counts())
    print("test label 分布：")
    print(test_df["label"].value_counts())

    train_df.to_csv(TRAIN_PATH, index=False, encoding="utf-8-sig")
    val_df.to_csv(VAL_PATH, index=False, encoding="utf-8-sig")
    test_df.to_csv(TEST_PATH, index=False, encoding="utf-8-sig")

    print("数据保存完成：")
    print(TRAIN_PATH)
    print(VAL_PATH)
    print(TEST_PATH)


if __name__ == "__main__":
    main()