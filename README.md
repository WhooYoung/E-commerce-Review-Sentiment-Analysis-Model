# 电商评论情感分析模型（Baseline + Longformer 整合版）

## 1. 项目简介

本项目整合了两条训练路线：

- 短文本路线：`baseline` / `transformer`（用于快速跑通与对比）
- 长文本路线：`Longformer`（用于长评论建模）

目标能力包括：

- 二分类情感识别（negative / positive）
- 单条评论预测
- Excel 批量预测
- Gradio 可视化演示

---

## 2. 版本与环境要求

推荐环境（Windows + conda）：

- OS：Windows 10 / 11
- Python：3.9 或 3.10（推荐 3.10）
- PyTorch：与本机 CUDA 对应版本
- Transformers：4.x
- Datasets：2.x

建议创建环境：

```bash
conda create -n ecommerce_sa python=3.10 -y
conda activate ecommerce_sa
```

安装依赖：

```bash
pip install -U pip setuptools wheel
pip install -r requirements.txt
```

---

## 3. 项目结构与功能说明

```text
E-commerce Review Sentiment Analysis Model/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ outputs/
│  ├─ figures/
│  └─ excel_results/
├─ app/
├─ config.py
├─ check_env.py
├─ prepare_data.py
├─ dataset.py
├─ models.py
├─ train_baseline.py
├─ train_longformer.py
├─ evaluate.py
├─ predict.py
├─ batch_infer_excel.py
├─ app_gradio.py
├─ requirements.txt
└─ README.md
```

### 3.1 目录作用

- `data/`
  - `data/processed/`：训练/验证/测试数据（含短文本和长文本拆分文件）
- `outputs/`
  - 存放模型权重、训练日志、Excel 推理结果
- `app/`
  - 预留应用层目录（当前演示入口是 `app_gradio.py`）

### 3.2 主要脚本作用

- `config.py`：统一管理路径、模型名、超参数、设备配置
- `check_env.py`：检查 Python / Torch / CUDA / Transformers 环境
- `prepare_data.py`：准备并保存 `train.csv`、`val.csv`、`test.csv`
- `dataset.py`：自定义数据集，支持 Longformer 的 `global_attention_mask`
- `models.py`：提供 `build_baseline()`、`build_longformer()`
- `train_baseline.py`：训练短文本 baseline，保存最优模型
- `train_longformer.py`：训练 Longformer，保存最优模型
- `evaluate.py`：输出 `classification_report` 与 `confusion_matrix`
- `predict.py`：单条评论推理（返回类别与概率）
- `batch_infer_excel.py`：读取 Excel 的 `text` 列并批量推理
- `app_gradio.py`：可视化界面，支持 baseline / longformer 切换

---

## 4. 数据与模型文件说明（当前整合状态）

### 4.1 数据文件

`data/processed/` 下包含：

- 短文本：`train.csv`、`val.csv`、`test.csv`
- 长文本：`long_train.csv`、`long_val.csv`、`long_test.csv`

### 4.2 模型与日志文件

`outputs/` 下包含（可能同时存在多套命名）：

- baseline：`best_baseline.pt` 或 `long_baseline.pt`
- longformer：`best_longformer.pt` 或 `long_longformer.pt`
- 训练日志：`baseline_train_log.csv`、`longformer_train_log.csv` 等

---

## 5. 执行步骤（建议顺序）

> 原则：先保证 baseline 可跑，再处理 Longformer。

### Step 1. 检查环境

```bash
python check_env.py
```

### Step 2. 准备数据

```bash
python prepare_data.py
```

### Step 3. 训练 baseline

```bash
python train_baseline.py
```

### Step 4. 评估 baseline

```bash
python evaluate.py --model_type baseline
```

### Step 5. 训练 Longformer

```bash
python train_longformer.py
```

### Step 6. 评估 Longformer

```bash
python evaluate.py --model_type longformer
```

### Step 7. 单条预测

```bash
python predict.py --model_type baseline --text "物流很快，包装很好，体验不错"
python predict.py --model_type longformer --text "物流很快，包装很好，体验不错"
```

### Step 8. Excel 批量预测

```bash
python batch_infer_excel.py --input sample.xlsx --output outputs/excel_results/result.xlsx --model_type baseline
```

> 输入 Excel 必须包含列名：`text`

### Step 9. 启动可视化界面

```bash
python app_gradio.py
```

---

## 6. 常见问题与处理

### 6.1 缺少依赖

报错：`ModuleNotFoundError`

处理：

```bash
pip install -r requirements.txt
```

### 6.2 Longformer 显存不足

报错：`CUDA out of memory`

处理：在 `config.py` 调小参数，例如：

```python
LONGFORMER_MAX_LEN = 512
LONGFORMER_BATCH_SIZE = 1
```

### 6.3 Longformer checkpoint 加载失败

可能是网络/模型兼容问题。处理建议：

1. 先跑通 baseline 全流程
2. 保留 Longformer 训练与推理代码框架
3. 在 `config.py` 中替换 `LONGFORMER_MODEL_NAME`
4. 重新执行 `train_longformer.py`

---

## 7. 课程设计答辩建议

- 先展示 baseline 流程（稳定、可复现）
- 再展示 Longformer 在长文本上的优势与限制


## 模型文件

由于GitHub对单文件大小有限制（100MB），大模型文件已从仓库中排除。

模型文件位置（本地）：
- `outputs/best_baseline.pt` - 基础模型（409MB）
- `outputs/best_longformer.pt` - Longformer模型（420MB）

运行项目时，请确保这些文件在`outputs/`文件夹中。
- 强调工程完整性：训练、评估、单条预测、批量预测、可视化演示全链路已打通

