"""D-FINE 火电厂目标检测验证脚本。

直接修改下方“用户验证参数配置区”的绝对路径后运行：

    python valid.py

输出 COCO 检测指标，并在 VALID_OUTPUT_DIR 保存 metrics.json 和 eval.pth。
"""

import copy
import csv
import json
import os
import time
from datetime import datetime
from pathlib import Path

# PyTorch 2.1 imports Transformers indirectly for ONNX helpers. D-FINE does
# not use Hugging Face models, so hide that irrelevant compatibility warning.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch

try:
    from calflops import calculate_flops
except ImportError:
    calculate_flops = None

from experiment_config import (
    MODEL_IMAGE_SIZE,
    MODEL_TAG,
    selected_model_config_path,
)
from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS
from src.solver.det_engine import evaluate


# =============================================================================
# 用户验证参数配置区：直接填写绝对路径
# =============================================================================
# 模型规模由 experiment_config.py 控制，改进模式由 my_improve/settings.py 控制；
# 两者必须与训练权重完全一致。
CONFIG_PATH = str(selected_model_config_path())

# 应用验证默认使用 mAP@0.5 最优的 best_map50.pth。也可改为
# best_map5095.pth、best_f1_fixed.pth 或 best_stg2.pth 进行同口径对比。
# 火电厂正式应用权重：CHECKPOINT_PATH = r"F:\3能源金三角基础设施识别\1模型推理应用可用权重\best_map50.pth"
CHECKPOINT_PATH = r"G:\b1\权重结果\改进实验dfine\dsqc_qcr\2e-4_SD3407\best_map50.pth"

# 验证影像目录和对应的 COCO JSON 标注文件。先运行 myscript/yolo2coco.py。
VAL_IMAGES_DIR = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\val\images")
VAL_ANNOTATION = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\val\annotations\val.json")
NUM_CLASSES = 1

# 必须与训练时的模型输入尺寸保持一致。
INPUT_SIZE = MODEL_IMAGE_SIZE
VAL_BATCH_SIZE = 6
NUM_WORKERS = 2
DEVICE = "cuda"
SEED = 3407

# 验证结果保存到单独目录，不覆盖训练过程中的输出文件。
VALID_OUTPUT_DIR = r"G:\b1\权重结果\改进实验dfine\dsqc_qcr\2e-4_SD3407\验证集精度"

# 只控制“论文指标.txt”中的显示名称，不参与模型结构或权重加载。
PAPER_MODEL_NAME = "dsqc_qcr_SD3407"

# 速度测试：单张 640×640 输入，先预热再重复计时。计时包含模型前向和检测后处理，
# 不含磁盘读取及 DataLoader 的图像变换，避免硬盘速度影响模型 FPS。
ENABLE_FPS_BENCHMARK = True
EFFICIENCY_CONFIDENCE = 0.50
FPS_WARMUP_ITERS = 10
FPS_TEST_ITERS = 100


# =============================================================================


COCO_METRIC_NAMES = [
    "mAP@0.5:0.95",
    "AP@0.5",
    "AP@0.75",
    "AP_small",
    "AP_medium",
    "AP_large",
    "AR@1",
    "AR@10",
    "AR@100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def format_elapsed_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS。"""
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def benchmark_single_image_fps(model, postprocessor, data_loader, device: torch.device) -> dict:
    """Measure batch-1 forward + postprocess + confidence filtering speed."""
    samples, targets = next(iter(data_loader))
    samples = samples[:1].to(device)
    target = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in targets[0].items()
    }
    orig_target_sizes = target["orig_size"].reshape(1, 2)

    def postprocess_at_efficiency_confidence(outputs):
        results = postprocessor(outputs, orig_target_sizes)
        for result in results:
            keep = result["scores"] >= EFFICIENCY_CONFIDENCE
            result["scores"] = result["scores"][keep]
            result["labels"] = result["labels"][keep]
            result["boxes"] = result["boxes"][keep]
        return results

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(FPS_WARMUP_ITERS):
            outputs = model(samples)
            postprocess_at_efficiency_confidence(outputs)
        synchronize()
        start = time.perf_counter()
        for _ in range(FPS_TEST_ITERS):
            outputs = model(samples)
            postprocess_at_efficiency_confidence(outputs)
        synchronize()
    seconds_per_image = (time.perf_counter() - start) / FPS_TEST_ITERS
    return {
        "FPS_single_image_forward_post": 1.0 / seconds_per_image,
        "latency_ms_single_image_forward_post": seconds_per_image * 1000.0,
        "warmup_iterations": FPS_WARMUP_ITERS,
        "test_iterations": FPS_TEST_ITERS,
        "efficiency_confidence": EFFICIENCY_CONFIDENCE,
    }


def calculate_model_flops(model) -> float:
    """使用 D-FINE 官方 calflops 口径统计 batch=1、当前输入尺寸的 GFLOPs。"""
    if calculate_flops is None:
        return 0.0

    profile_model = copy.deepcopy(dist_utils.de_parallel(model)).cpu().eval()
    if hasattr(profile_model, "deploy"):
        profile_model = profile_model.deploy()
    flops, _, _ = calculate_flops(
        model=profile_model,
        input_shape=(1, 3, INPUT_SIZE, INPUT_SIZE),
        print_results=False,
        print_detailed=False,
        output_as_string=False,
    )
    del profile_model
    return float(flops) / 1e9


def save_confidence_curve_outputs(result_dir: Path, confidence_metrics: dict) -> None:
    """Save editable threshold metrics and a confidence-vs-metrics figure."""
    curve = confidence_metrics.get("curve", [])
    if not curve:
        return

    csv_path = result_dir / "confidence_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve[0].keys()))
        writer.writeheader()
        writer.writerows(curve)

    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"matplotlib 未安装，已保存置信度 CSV: {csv_path}")
        return

    confidence = [item["confidence"] for item in curve]
    figure, axis = plt.subplots(figsize=(9, 6), dpi=180, constrained_layout=True)
    for key, label, color in (
        ("precision", "Precision", "#1f77b4"),
        ("recall", "Recall", "#ff7f0e"),
        ("f1", "F1", "#2ca02c"),
    ):
        axis.plot(confidence, [item[key] for item in curve], label=label, color=color)
    recommended = confidence_metrics["recommended_confidence"]
    axis.axvline(recommended, color="#d62728", linestyle="--", label=f"Best F1 @ {recommended:.2f}")
    axis.set_xlabel("Confidence threshold")
    axis.set_ylabel("Metric")
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.28, linestyle="--")
    axis.legend()
    figure.savefig(result_dir / "confidence_precision_recall_f1.png", bbox_inches="tight")
    plt.close(figure)


def save_paper_metrics(
    result_dir: Path,
    coco_metrics: dict,
    threshold_metrics: dict,
    speed_metrics: dict,
    model_params: int,
    flops_g: float,
) -> Path:
    """生成可直接复制到论文或表格软件中的精度表和效率表。"""
    latency = speed_metrics.get("latency_ms_single_image_forward_post")
    fps = speed_metrics.get("FPS_single_image_forward_post")
    flops_text = f"{flops_g:.3f}" if flops_g > 0 else "N/A"
    latency_text = f"{latency:.4f}" if latency is not None else "N/A"
    fps_text = f"{fps:.4f}" if fps is not None else "N/A"

    accuracy_header = (
        f"{'Model':<16}{'Input':>8}{'Params(M)':>14}{'AP50':>12}{'AP75':>12}"
        f"{'mAP50:95':>14}{'P@0.5':>12}{'R@0.5':>12}{'F1@0.5':>12}"
    )
    accuracy_row = (
        f"{PAPER_MODEL_NAME:<16}{INPUT_SIZE:>8}{model_params / 1e6:>14.3f}"
        f"{coco_metrics['AP@0.5']:>12.4f}{coco_metrics['AP@0.75']:>12.4f}"
        f"{coco_metrics['mAP@0.5:0.95']:>14.4f}"
        f"{threshold_metrics['precision']:>12.4f}{threshold_metrics['recall']:>12.4f}"
        f"{threshold_metrics['f1']:>12.4f}"
    )
    efficiency_header = (
        f"{'Model':<16}{'Input':>8}{'Params(M)':>14}{'FLOPs(G)':>14}"
        f"{'Latency(ms/image)':>22}{'FPS':>14}"
    )
    efficiency_row = (
        f"{PAPER_MODEL_NAME:<16}{INPUT_SIZE:>8}{model_params / 1e6:>14.3f}"
        f"{flops_text:>14}{latency_text:>22}{fps_text:>14}"
    )

    lines = [
        "D-FINE 论文对比实验指标",
        "=" * 104,
        "一、精度对比表",
        accuracy_header,
        "-" * len(accuracy_header),
        accuracy_row,
        "",
        "二、效率对比表",
        efficiency_header,
        "-" * len(efficiency_header),
        efficiency_row,
        "",
        "评价口径：",
        "1. P@0.5、R@0.5、F1@0.5：置信度≥0.50，匹配IoU≥0.50。",
        "2. AP50、AP75、mAP50:95：COCO标准检测评价。",
        f"3. FLOPs：D-FINE官方calflops部署图口径，batch=1，输入{INPUT_SIZE}×{INPUT_SIZE}。",
        f"4. Latency与FPS：batch=1，置信度={EFFICIENCY_CONFIDENCE:.2f}，模型前向+检测后处理。",
        f"5. 测速：预热{FPS_WARMUP_ITERS}次，正式测试{FPS_TEST_ITERS}次，不含磁盘读取和DataLoader变换。",
        "",
        "制表提示：以上列使用固定宽度排版；也可按列复制到 Word 或 Excel。",
    ]
    path = result_dir / "论文指标.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_metrics_txt(
    path: Path,
    coco_metrics: dict,
    threshold_metrics: dict,
    confidence_metrics: dict,
    speed_metrics: dict,
    model_params: int,
    flops_g: float,
    started_at: datetime,
    elapsed_seconds: float,
) -> None:
    """Write a human-readable report including the metric definitions used in the paper."""
    lines = [
        "火电厂目标检测验证精度报告",
        "=" * 48,
        f"模型权重: {CHECKPOINT_PATH}",
        f"验证影像: {VAL_IMAGES_DIR}",
        f"验证标注: {VAL_ANNOTATION}",
        f"输入尺寸: {INPUT_SIZE} × {INPUT_SIZE}",
        "",
        "一、与 YOLO11-DAE 论文一致的核心指标",
        "说明：P、R、F1 使用置信度阈值 0.50、IoU 阈值 0.50 的匹配结果；mAP 使用 COCO 标准定义。",
        f"Precision (P)                 : {threshold_metrics.get('precision', float('nan')):.4f}",
        f"Recall (R)                    : {threshold_metrics.get('recall', float('nan')):.4f}",
        f"F1-score                      : {threshold_metrics.get('f1', float('nan')):.4f}",
        f"mAP@0.5                       : {coco_metrics.get('AP@0.5', float('nan')):.4f}",
        f"推荐最佳置信度              : {confidence_metrics.get('recommended_confidence', float('nan')):.2f}",
        f"该置信度下 F1               : {confidence_metrics.get('best_f1_metrics', {}).get('f1', float('nan')):.4f}",
        "",
        "二、补充 COCO 检测精度指标",
    ]
    lines.extend(f"{name:30s}: {value:.4f}" for name, value in coco_metrics.items())
    lines.extend(
        [
            "",
            "三、阈值检测统计（置信度≥0.50，IoU≥0.50）",
        ]
    )
    lines.extend(f"{name:30s}: {value:.4f}" for name, value in threshold_metrics.items())
    lines.extend(
        [
            "",
            "四、效率参考指标",
            f"模型参数量 (M)                 : {model_params / 1e6:.3f}",
            f"FLOPs (G)                       : {flops_g:.3f}"
            if flops_g > 0
            else "FLOPs (G)                       : N/A（未安装 calflops）",
            f"效率测试置信度                  : {EFFICIENCY_CONFIDENCE:.2f}",
        ]
    )
    lines.extend(f"{name:30s}: {value:.4f}" for name, value in speed_metrics.items())
    lines.extend(
        [
            "",
            "五、计算公式与符号说明",
            "P = TP / (TP + FP)",
            "R = TP / (TP + FN)",
            "F1 = 2 × P × R / (P + R)",
            "IoU = Area(B_pred ∩ B_gt) / Area(B_pred ∪ B_gt)",
            "AP = ∫_0^1 P(R) dR",
            "mAP@0.5 = (1 / N) × Σ AP_i，IoU 阈值固定为 0.50",
            "mAP@0.5:0.95 = (1 / 10N) × Σ_(t∈{0.50,0.55,...,0.95}) Σ_i AP_i(t)",
            "FPS = 1 / T_image；论文写法等价于 FPS = 1000 / (Pre + Infer + Post)。",
            "其中 TP、FP、FN 分别表示正确检出、误检和漏检数量；N 为类别数。本研究 N=1（火电厂）。",
            "",
            "速度说明：本报告 FPS 为单张图像的“模型前向 + 检测后处理”速度，不含磁盘读取和 DataLoader 变换。",
            "若与其他论文比较 FPS，必须保持显卡、PyTorch/CUDA、输入尺寸、批量大小和计时范围一致。",
            "",
            "六、验证耗时",
            f"开始时间：{started_at.isoformat(timespec='seconds')}",
            f"结束时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"总耗时：{format_elapsed_time(elapsed_seconds)}",
            f"总秒数：{elapsed_seconds:.2f}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def replace_resize_size(ops, input_size: int) -> None:
    """使验证 Resize 与 INPUT_SIZE 同步。"""
    for op in ops or []:
        if isinstance(op, dict) and op.get("type") == "Resize":
            op["size"] = [input_size, input_size]


def build_validation_config() -> YAMLConfig:
    """读取模型 YAML，并用顶部参数覆盖验证相关设置。"""
    overrides = yaml_utils.load_config(CONFIG_PATH)
    overrides["num_classes"] = NUM_CLASSES
    overrides["remap_mscoco_category"] = False
    overrides["device"] = DEVICE
    overrides["sync_bn"] = False
    overrides["output_dir"] = VALID_OUTPUT_DIR
    overrides["eval_spatial_size"] = [INPUT_SIZE, INPUT_SIZE]
    overrides["resume"] = CHECKPOINT_PATH
    overrides["tuning"] = None

    val_loader = overrides["val_dataloader"]
    val_loader["total_batch_size"] = VAL_BATCH_SIZE
    val_loader["num_workers"] = NUM_WORKERS
    val_loader["dataset"]["img_folder"] = VAL_IMAGES_DIR
    val_loader["dataset"]["ann_file"] = VAL_ANNOTATION
    replace_resize_size(val_loader["dataset"]["transforms"].get("ops"), INPUT_SIZE)

    return YAMLConfig(CONFIG_PATH, **overrides)


def validate_paths() -> None:
    """在模型加载前检查用户填写的路径。"""
    required = {
        "模型配置": CONFIG_PATH,
        "模型权重": CHECKPOINT_PATH,
        "验证影像目录": VAL_IMAGES_DIR,
        "验证标注": VAL_ANNOTATION,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("以下路径不存在：\n" + "\n".join(missing))
    if INPUT_SIZE <= 0 or INPUT_SIZE % 32 != 0:
        raise ValueError("INPUT_SIZE 必须为能被 32 整除的正整数。")
    if VAL_BATCH_SIZE <= 0:
        raise ValueError("VAL_BATCH_SIZE 必须大于 0。")
    if FPS_WARMUP_ITERS < 0 or FPS_TEST_ITERS <= 0:
        raise ValueError("FPS_WARMUP_ITERS 必须≥0，FPS_TEST_ITERS 必须>0。")
    if not 0 <= EFFICIENCY_CONFIDENCE <= 1:
        raise ValueError("EFFICIENCY_CONFIDENCE 必须在 [0, 1] 范围内。")
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE 设置为 cuda，但当前没有可用 CUDA 显卡。")


def main() -> None:
    validate_paths()
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()
    print("\n========== 火电厂目标检测验证 ==========")
    print(f"共享模型选择: {MODEL_TAG}")
    print(f"模型配置: {CONFIG_PATH}")
    print(f"验证权重: {CHECKPOINT_PATH}")
    print(f"验证集: {VAL_IMAGES_DIR}")
    print(f"输入尺寸: {INPUT_SIZE} | 验证批量: {VAL_BATCH_SIZE} | 设备: {DEVICE}")
    print(f"结果目录: {VALID_OUTPUT_DIR}\n")

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=SEED)
    cfg = build_validation_config()

    # CHECKPOINT_PATH already contains the complete trained model.  Disable the
    # separate HGNetV2 backbone download before solver.eval() loads it.
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    solver = TASKS[cfg.yaml_cfg["task"]](cfg)

    # eval() 会加载 CHECKPOINT_PATH 中完整的 model 与 EMA 状态。
    solver.eval()
    model = solver.ema.module if solver.ema else solver.model
    # 使用模型副本计算FLOPs，不改变已加载权重及后续正式验证模型。
    flops_g = calculate_model_flops(model)
    stats, coco_evaluator = evaluate(
        model=model,
        criterion=solver.criterion,
        postprocessor=solver.postprocessor,
        data_loader=solver.val_dataloader,
        coco_evaluator=solver.evaluator,
        device=solver.device,
        epoch=-1,
        use_wandb=False,
        return_detection_metrics=True,
        return_confidence_metrics=True,
    )

    coco_metrics = dict(zip(COCO_METRIC_NAMES, stats["coco_eval_bbox"]))
    threshold_metrics = {
        name: float(value) for name, value in stats.get("detection_metrics", {}).items()
    }
    confidence_metrics = stats["confidence_metrics"]
    model_params = sum(parameter.numel() for parameter in model.parameters())
    speed_metrics = (
        benchmark_single_image_fps(model, solver.postprocessor, solver.val_dataloader, solver.device)
        if ENABLE_FPS_BENCHMARK
        else {}
    )
    elapsed_seconds = time.perf_counter() - start_time
    print("\n========== 最终验证结果 ==========")
    for name, value in coco_metrics.items():
        print(f"{name:16s}: {value:.4f}")
    print("\n========== 阈值检测指标 ==========")
    for name, value in threshold_metrics.items():
        print(f"{name:16s}: {value:.4f}")
    best_f1_metrics = confidence_metrics["best_f1_metrics"]
    print("\n========== 验证集最佳 F1 工作点 ==========")
    print(f"推荐置信度    : {confidence_metrics['recommended_confidence']:.2f}")
    print(f"Precision       : {best_f1_metrics['precision']:.4f}")
    print(f"Recall          : {best_f1_metrics['recall']:.4f}")
    print(f"F1-score        : {best_f1_metrics['f1']:.4f}")
    print(
        f"TP / FP / FN    : {best_f1_metrics['TPs']} / "
        f"{best_f1_metrics['FPs']} / {best_f1_metrics['FNs']}"
    )
    print("\n========== 论文速度 / 规模参考指标 ==========")
    print(f"模型参数量 (M)    : {model_params / 1e6:.3f}")
    print(f"FLOPs (G)         : {flops_g:.3f}" if flops_g > 0 else "FLOPs (G)         : N/A")
    for name, value in speed_metrics.items():
        print(f"{name:30s}: {value:.4f}")
    print(f"验证总耗时          : {format_elapsed_time(elapsed_seconds)}")

    result_dir = Path(VALID_OUTPUT_DIR)
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "metrics.json").write_text(
        json.dumps(
            {
                "paper_core_metrics": {
                    "Precision": threshold_metrics.get("precision"),
                    "Recall": threshold_metrics.get("recall"),
                    "F1-score": threshold_metrics.get("f1"),
                    "mAP@0.5": coco_metrics.get("AP@0.5"),
                },
                "coco_metrics": coco_metrics,
                "threshold_metrics": threshold_metrics,
                "confidence_metrics": confidence_metrics,
                "speed_metrics": speed_metrics,
                "model_parameters": model_params,
                "FLOPs_G": flops_g if flops_g > 0 else None,
                "efficiency_confidence": EFFICIENCY_CONFIDENCE,
                "validation_elapsed_seconds": elapsed_seconds,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_confidence_curve_outputs(result_dir, confidence_metrics)
    paper_path = save_paper_metrics(
        result_dir,
        coco_metrics,
        threshold_metrics,
        speed_metrics,
        model_params,
        flops_g,
    )
    txt_path = result_dir / "精度指标报告.txt"
    write_metrics_txt(
        txt_path,
        coco_metrics,
        threshold_metrics,
        confidence_metrics,
        speed_metrics,
        model_params,
        flops_g,
        started_at,
        elapsed_seconds,
    )
    torch.save(coco_evaluator.coco_eval["bbox"].eval, result_dir / "eval.pth")
    print(f"\n指标已保存：{result_dir / 'metrics.json'}")
    print(f"TXT 精度报告已保存：{txt_path}")
    print(f"论文指标已保存：{paper_path}")
    print(f"COCO 评估对象已保存：{result_dir / 'eval.pth'}")
    dist_utils.cleanup()


if __name__ == "__main__":
    main()
