"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import csv
import json
import os
import sys
import time

# PyTorch 2.1 probes an installed Transformers package while importing ONNX
# helpers.  D-FINE does not use Hugging Face models, so suppress that unrelated
# compatibility warning before importing torch.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse

from experiment_config import (
    ACTIVE_IMPROVEMENT_MODE,
    MODEL_IMAGE_SIZE,
    MODEL_TAG,
    PRETRAINED_WEIGHT_PATH,
    USE_MGCA,
    USE_QACG,
    USE_DSQC,
    USE_QFBCG,
    USE_QLCS,
    USE_SHEA,
    USE_QCR,
    USE_VRAC_AUGMENTATION,
    selected_model_config_path,
)
from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS

# =============================================================================
# 用户训练参数配置区（直接运行 `python train.py` 时使用）
# =============================================================================
# 推荐：RTX 4060 Laptop 8GB 优先使用 D-FINE-S。可选值为 "s"、"m"、"l"、"x"；
# L/X 显存占用明显更高，切换后如显存不足，应优先减小训练批量。
# S/M/L/X 规模及 VRAC 在 experiment_config.py 中设置；网络改进开关统一放在
# my_improve/settings.py，训练、验证和测试会同步选择相同 YAML。

# COCO 格式数据集：直接填写绝对路径。Windows 路径前请保留 r，避免反斜杠被转义。
# 训练/验证数据的绝对路径。训练前先运行 myscript/yolo2coco.py 生成 JSON。
TRAIN_IMAGES_DIR = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\train\images")
TRAIN_ANNOTATION = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\train\annotations\train.json")
VAL_IMAGES_DIR = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\val\images")
VAL_ANNOTATION = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\val\annotations\val.json")
NUM_CLASSES = 1  # 当前仅检测火电厂（hdc）

# 8GB 单卡推荐起点。若显存不足改为 2；显存余量较大可尝试 6 或 8。
INPUT_SIZE = MODEL_IMAGE_SIZE
TRAIN_BATCH_SIZE = 6
VAL_BATCH_SIZE = 6
NUM_WORKERS = 2  # Windows 上建议 0~2；若 DataLoader 异常可改为 0
EPOCHS = 100
# 前 180 轮使用颜色、模糊、噪声、外扩和 VRAC，最后 20 轮用干净样本稳定收敛。
AUGMENTATION_STOP_EPOCH = 90

#EPOCHS和AUGMENTATION_STOP_EPOCH的关系：                       EPOCHS=200时，AUGMENTATION_STOP_EPOCH=180；
#EPOCHS=40时，AUGMENTATION_STOP_EPOCH=36；                      EPOCHS = 240     150
#正式论文中应用EPOCHS=200、AUGMENTATION_STOP_EPOCH=180；         AUGMENTATION_STOP_EPOCH = 210      135
#快速测试时应用EPOCHS=40、AUGMENTATION_STOP_EPOCH=36。

# 训练结果输出目录：每次 S/M/L/X、基线/改进实验请填写独立绝对路径。
OUTPUT_DIR = r"E:\YOLO\D-FINE\output\qlcs_dsqc_qcr\2e-4_SD3407"

SEED = 3407                      #18、2026、3407
DEVICE = "cuda"
USE_AMP = True

# S 模型原配置的学习率为 4e-4。修改 BASE_LR 时，骨干网络学习率会保持为其 0.5 倍。
# M模型建议改为BASE_LR = 2e-4

BASE_LR = 2e-4
WEIGHT_DECAY = 1e-4
PRINT_FREQ = 20
CHECKPOINT_FREQ = 10


# 由 experiment_config.py 按 S/M/L/X 自动选择 weight 目录中的 COCO 预训练权重。
# 留空时仍会使用 HGNetv2 预训练骨干，但不是完整检测器预训练权重。
TUNING_CHECKPOINT = str(PRETRAINED_WEIGHT_PATH)

# 训练中断后续训时填写 last.pth；续训与预训练微调不能同时启用。
# 示例：OUTPUT_DIR / "last.pth"
RESUME_CHECKPOINT = ""


debug = False

if debug:
    def custom_repr(self):
        return f"{{Tensor:{tuple(self.shape)}}} {original_repr(self)}"

    original_repr = torch.Tensor.__repr__
    torch.Tensor.__repr__ = custom_repr


def safe_get_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank()
    else:
        return 0


def get_default_config_path() -> Path:
    """根据顶部参数区选择模型配置文件。"""
    return selected_model_config_path()


def _replace_resize_size(ops, input_size: int) -> None:
    """把训练流水线中的 Resize 同步为用户设置的输入尺寸。"""
    for op in ops or []:
        if isinstance(op, dict) and op.get("type") == "Resize":
            op["size"] = [input_size, input_size]


def build_user_overrides(config_path: str) -> dict:
    """把顶部用户参数转换为仓库原生 YAML 配置覆盖项。"""
    cfg = yaml_utils.load_config(config_path)

    cfg["num_classes"] = NUM_CLASSES
    cfg["remap_mscoco_category"] = False
    cfg["epochs"] = EPOCHS
    cfg["use_amp"] = USE_AMP
    cfg["sync_bn"] = False  # 当前默认是单卡训练
    cfg["print_freq"] = PRINT_FREQ
    cfg["checkpoint_freq"] = CHECKPOINT_FREQ
    cfg["output_dir"] = str(OUTPUT_DIR)
    cfg["eval_spatial_size"] = [INPUT_SIZE, INPUT_SIZE]

    train_loader = cfg["train_dataloader"]
    train_loader["total_batch_size"] = TRAIN_BATCH_SIZE
    train_loader["num_workers"] = NUM_WORKERS
    train_loader["dataset"]["img_folder"] = str(TRAIN_IMAGES_DIR)
    train_loader["dataset"]["ann_file"] = str(TRAIN_ANNOTATION)
    train_transforms = train_loader["dataset"]["transforms"]
    _replace_resize_size(train_transforms.get("ops"), INPUT_SIZE)
    if train_transforms.get("policy", {}).get("name") == "stop_epoch":
        train_transforms["policy"]["epoch"] = AUGMENTATION_STOP_EPOCH
    train_loader["collate_fn"]["base_size"] = INPUT_SIZE
    train_loader["collate_fn"]["stop_epoch"] = AUGMENTATION_STOP_EPOCH

    val_loader = cfg["val_dataloader"]
    val_loader["total_batch_size"] = VAL_BATCH_SIZE
    val_loader["num_workers"] = NUM_WORKERS
    val_loader["dataset"]["img_folder"] = str(VAL_IMAGES_DIR)
    val_loader["dataset"]["ann_file"] = str(VAL_ANNOTATION)
    _replace_resize_size(val_loader["dataset"]["transforms"].get("ops"), INPUT_SIZE)

    optimizer = cfg["optimizer"]
    original_base_lr = float(optimizer["lr"])
    for group in optimizer.get("params", []):
        if "lr" in group:
            group["lr"] = BASE_LR * float(group["lr"]) / original_base_lr
    optimizer["lr"] = BASE_LR
    optimizer["weight_decay"] = WEIGHT_DECAY
    return cfg


def validate_training_inputs(args) -> None:
    """在启动耗时的模型构建前给出清晰的配置错误。"""
    if INPUT_SIZE <= 0 or INPUT_SIZE % 32 != 0:
        raise ValueError("INPUT_SIZE 必须是正数且能被 32 整除。")
    if TRAIN_BATCH_SIZE <= 0 or VAL_BATCH_SIZE <= 0:
        raise ValueError("TRAIN_BATCH_SIZE 和 VAL_BATCH_SIZE 必须大于 0。")
    if not 0 <= AUGMENTATION_STOP_EPOCH <= EPOCHS:
        raise ValueError("AUGMENTATION_STOP_EPOCH 必须在 0 到 EPOCHS 之间。")
    required_paths = {
        "训练影像目录": TRAIN_IMAGES_DIR,
        "训练标注": TRAIN_ANNOTATION,
        "验证影像目录": VAL_IMAGES_DIR,
        "验证标注": VAL_ANNOTATION,
        "模型配置": Path(args.config),
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("以下训练路径不存在：\n" + "\n".join(missing))
    if args.tuning and not Path(args.tuning).is_file():
        raise FileNotFoundError(f"预训练权重不存在：{args.tuning}")
    if args.resume and not Path(args.resume).is_file():
        raise FileNotFoundError(f"续训权重不存在：{args.resume}")
    if args.tuning and args.resume:
        raise ValueError("TUNING_CHECKPOINT 与 RESUME_CHECKPOINT 只能填写一个。")
    if args.device and str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE 设置为 cuda，但当前 PyTorch 没有检测到可用的 CUDA 显卡。")


def print_user_config(args) -> None:
    """启动时先打印最常核对的参数，避免用错数据或权重。"""
    print("\n========== 火电厂目标检测训练参数 ==========")
    print(f"模型: {MODEL_TAG}")
    print(
        f"QLCS模块: {'开启' if USE_QLCS else '关闭'} | "
        f"QFBCG模块: {'开启' if USE_QFBCG else '关闭'} | "
        f"DSQC模块: {'开启' if USE_DSQC else '关闭'} | "
        f"QACG模块: {'开启' if USE_QACG else '关闭'} | "
        f"MGCA模块: {'开启' if USE_MGCA else '关闭'} | "
        f"SHEA模块: {'开启' if USE_SHEA else '关闭'} | "
        f"QCR训练正则: {'开启' if USE_QCR else '关闭'} | "
        f"网络改进模式: {ACTIVE_IMPROVEMENT_MODE} | "
        f"VRAC增强: {'开启' if USE_VRAC_AUGMENTATION else '关闭'}"
    )
    print(f"训练集: {TRAIN_IMAGES_DIR}")
    print(f"验证集: {VAL_IMAGES_DIR}")
    print(
        f"输入尺寸: {INPUT_SIZE} | 训练批量: {TRAIN_BATCH_SIZE} | "
        f"轮数: {EPOCHS} | 增强停止轮次: {AUGMENTATION_STOP_EPOCH}"
    )
    print(f"设备: {args.device} | AMP: {USE_AMP if args.use_amp is None else args.use_amp}")
    print(f"预训练微调: {args.tuning or '未设置（仅使用预训练骨干）'}")
    print(f"断点续训: {args.resume or '未设置'}")
    print(f"输出目录: {args.output_dir}\n")


def format_elapsed_seconds(seconds: float) -> str:
    """Format elapsed seconds as HH:MM:SS with the exact duration in seconds."""
    whole_seconds = int(seconds)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d} ({seconds:.2f} 秒)"


def export_training_curves(output_dir: Path) -> None:
    """Parse D-FINE log.txt and export paper-ready convergence curves and CSV.

    The detector writes one JSON object per epoch.  COCO bbox statistics follow
    the official order: [mAP50:95, AP50, AP75, APsmall, APmedium, APlarge,
    AR1, AR10, AR100, ARsmall, ARmedium, ARlarge].
    """
    log_path = output_dir / "log.txt"
    if not log_path.is_file():
        print(f"[训练曲线] 未找到日志，跳过绘图：{log_path}")
        return

    history = []
    for line_number, line in enumerate(log_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            metrics = record.get("test_coco_eval_bbox")
            detection = record.get("test_detection_metrics", {})
            if not isinstance(metrics, list) or len(metrics) < 9:
                continue
            history.append(
                {
                    "epoch": int(record["epoch"]),
                    "train_loss": float(record.get("train_loss", "nan")),
                    "lr": float(record.get("train_lr", "nan")),
                    "mAP@0.5:0.95": float(metrics[0]),
                    "mAP@0.5": float(metrics[1]),
                    "AP@0.75": float(metrics[2]),
                    "AR@100": float(metrics[8]),
                    "Precision@conf0.5": float(detection.get("precision", "nan")),
                    "Recall@conf0.5": float(detection.get("recall", "nan")),
                    "F1@conf0.5": float(detection.get("f1", "nan")),
                }
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            print(f"[训练曲线] 跳过 log.txt 第 {line_number} 行：{error}")

    history.sort(key=lambda item: item["epoch"])
    if len(history) < 2:
        print("[训练曲线] 有效 epoch 记录少于 2 条，暂不生成曲线。")
        return

    csv_path = output_dir / "training_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"[训练曲线] matplotlib 未安装，已保存 CSV：{csv_path}")
        return

    epochs = [item["epoch"] for item in history]
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180, constrained_layout=True)
    plots = [
        ("train_loss", "Training loss", "#1f77b4"),
        ("mAP@0.5", "mAP@0.5", "#2ca02c"),
        ("mAP@0.5:0.95", "mAP@0.5:0.95", "#d62728"),
        ("F1@conf0.5", "F1 @ confidence 0.5", "#9467bd"),
    ]
    for axis, (key, title, color) in zip(axes.flat, plots):
        axis.plot(epochs, [item[key] for item in history], color=color, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(key)
        axis.grid(alpha=0.28, linestyle="--")
        # 标示停止在线增强、转入干净样本收敛的 epoch，方便解释曲线阶段变化。
        if epochs[0] <= AUGMENTATION_STOP_EPOCH <= epochs[-1]:
            axis.axvline(AUGMENTATION_STOP_EPOCH, color="#555555", linestyle="--", linewidth=1)
            axis.text(
                AUGMENTATION_STOP_EPOCH,
                axis.get_ylim()[1],
                " augmentation off",
                color="#555555",
                fontsize=8,
                va="top",
            )

    figure.suptitle("D-FINE-S Training Convergence Curves", fontsize=14)
    curve_path = output_dir / "training_convergence_curves.png"
    figure.savefig(curve_path, bbox_inches="tight")
    plt.close(figure)

    print("\n========== 训练曲线已生成 ==========")
    print(f"CSV：{csv_path}")
    print(f"曲线图：{curve_path}")


def main(args) -> None:
    """main"""
    # 不训练，只根据已有 log.txt 补生成曲线；适用于已完成的历史实验。
    if args.plot_only:
        export_training_curves(Path(args.output_dir))
        return

    total_start_time = time.perf_counter()
    validate_training_inputs(args)
    print_user_config(args)
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    assert not all(
        [args.tuning, args.resume]
    ), "Only support from_scrach or resume or tuning at one time"

    update_dict = build_user_overrides(args.config)
    # 命令行 -u 的优先级高于顶部用户配置区。
    update_dict = yaml_utils.merge_dict(update_dict, yaml_utils.parse_cli(args.update))
    update_dict.update(
        {
            k: v
            for k, v in args.__dict__.items()
            if k
            not in [
                "update",
                "config",
            ]
            and v is not None
        }
    )

    cfg = YAMLConfig(args.config, **update_dict)

    if args.resume or args.tuning:
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    if args.test_only:
        solver.val()
    else:
        solver.fit()

    # Record complete wall-clock duration, including setup and validation.
    total_elapsed = time.perf_counter() - total_start_time
    if safe_get_rank() == 0:
        elapsed_text = format_elapsed_seconds(total_elapsed)
        print("\n========== 任务总耗时 ==========")
        print(elapsed_text)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        duration_file = output_dir / (
            "validation_time.txt" if args.test_only else "training_time.txt"
        )
        duration_file.write_text(
            "D-FINE 训练/验证耗时记录\n"
            f"模式: {'仅验证' if args.test_only else '训练'}\n"
            f"总耗时: {elapsed_text}\n",
            encoding="utf-8",
        )
        print(f"耗时记录已保存: {duration_file}")

        # 每次完整训练结束后，自动从 log.txt 输出论文常用收敛曲线与可编辑 CSV。
        if not args.test_only:
            export_training_curves(output_dir)

    dist_utils.cleanup()


if __name__ == "__main__":
    default_config = get_default_config_path()
    parser = argparse.ArgumentParser(
        description="D-FINE 火电厂目标检测训练（默认参数可在 train.py 顶部修改）"
    )

    # priority 0
    parser.add_argument(
        "-c", "--config", type=str, default=str(default_config),
        help="YAML config; direct execution also follows experiment_config.py and my_improve/settings.py",
    )
    parser.add_argument(
        "-r", "--resume", type=str, default=str(RESUME_CHECKPOINT) or None,
        help="resume complete training state from checkpoint",
    )
    parser.add_argument(
        "-t", "--tuning", type=str, default=str(TUNING_CHECKPOINT) or None,
        help="fine-tune from pretrained model weights",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default=DEVICE,
        help="device",
    )
    parser.add_argument("--seed", type=int, default=SEED, help="experiment reproducibility")
    parser.add_argument(
        "--use-amp", dest="use_amp", action="store_true", default=None,
        help="enable automatic mixed precision training",
    )
    parser.add_argument(
        "--no-amp", dest="use_amp", action="store_false",
        help="disable automatic mixed precision training",
    )
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="output directory")
    parser.add_argument("--summary-dir", type=str, help="tensorboard summry")
    parser.add_argument(
        "--test-only",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=False,
        help="不训练；仅根据 output_dir/log.txt 生成训练收敛曲线和 CSV",
    )

    # priority 1
    parser.add_argument("-u", "--update", nargs="+", help="update yaml config")

    # env
    parser.add_argument("--print-method", type=str, default="builtin", help="print method")
    parser.add_argument("--print-rank", type=int, default=0, help="print rank id")

    parser.add_argument("--local-rank", type=int, help="local rank id")
    args = parser.parse_args()

    main(args)
