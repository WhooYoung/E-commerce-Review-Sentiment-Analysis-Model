import platform

import torch


def main() -> None:
    try:
        import transformers  # type: ignore
    except Exception as e:  # pragma: no cover - 环境检查脚本不做严格单元测试
        transformers = None
        print("导入 transformers 失败：", e)

    print("===== 环境检查结果 =====")
    print(f"Python 版本: {platform.python_version()}")
    print(f"操作系统: {platform.system()} {platform.version()}")

    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        try:
            print(f"GPU 数量: {torch.cuda.device_count()}")
            print(f"当前 GPU 名称: {torch.cuda.get_device_name(0)}")
        except Exception as e:
            print("获取 GPU 信息失败：", e)

    if transformers is not None:
        print(f"Transformers 版本: {transformers.__version__}")
    else:
        print("Transformers 未正确安装，请先执行: pip install transformers")


if __name__ == "__main__":
    main()

