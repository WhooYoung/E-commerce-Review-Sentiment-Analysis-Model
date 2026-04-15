import os
import torch


# 项目根目录（自动获取当前文件所在目录的上级）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 数据与输出目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
FIGURE_DIR = os.path.join(OUTPUT_DIR, "figures")
EXCEL_RESULT_DIR = os.path.join(OUTPUT_DIR, "excel_results")

# 模型与日志文件路径
BEST_BASELINE_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_baseline.pt")
BEST_LONGFORMER_MODEL_PATH = os.path.join(OUTPUT_DIR, "best_longformer.pt")

BASELINE_TRAIN_LOG = os.path.join(OUTPUT_DIR, "baseline_train_log.csv")
LONGFORMER_TRAIN_LOG = os.path.join(OUTPUT_DIR, "longformer_train_log.csv")

# 预训练模型名称
BASELINE_MODEL_NAME = "bert-base-chinese"

# Longformer 模型名称，如果该 checkpoint 无法加载，可在此处替换
LONGFORMER_MODEL_NAME = "schen/longformer-chinese-base-4096"

# 数据集名称
HF_DATASET_NAME = "ttxy/online_shopping_10_cats"

# 训练超参数（可根据显存情况在此统一调节）
BASELINE_MAX_LEN = 256
BASELINE_BATCH_SIZE = 16
BASELINE_EPOCHS = 2
BASELINE_LR = 2e-5

LONGFORMER_MAX_LEN = 1024  # 如显存不足，可在 README / 本文件中调小
LONGFORMER_BATCH_SIZE = 2
LONGFORMER_EPOCHS = 2
LONGFORMER_LR = 2e-5

# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dirs() -> None:
    """
    确保所有需要的目录都已经创建。
    在训练、评估、预测等脚本开头调用一次即可。
    """
    for path in [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        OUTPUT_DIR,
        FIGURE_DIR,
        EXCEL_RESULT_DIR,
    ]:
        os.makedirs(path, exist_ok=True)

