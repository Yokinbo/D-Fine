"""D-FINE 火电厂目标检测独立测试集评估脚本。

用途：模型结构、训练轮数、最佳权重和置信度口径在验证集上全部确定后，
仅使用独立 test 集生成论文最终核心对比指标。

直接修改下方“用户测试参数配置区”，然后在仓库根目录运行：

    python test.py

注意：本脚本故意不在测试集上搜索最佳置信度。P、R、F1 固定采用
置信度 0.50、匹配 IoU 0.50；AP 指标采用 COCO 标准评价。
"""

from __future__ import annotations

import copy
import json
import os
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import torch

try:
    from calflops import calculate_flops
except ImportError:
    calculate_flops = None

from experiment_config import MODEL_IMAGE_SIZE, MODEL_TAG, selected_model_config_path
from src.core import YAMLConfig, yaml_utils
from src.misc import dist_utils
from src.solver import TASKS
from src.solver.det_engine import evaluate


# =============================================================================
# 用户测试参数配置区
# =============================================================================

# 模型规模由 experiment_config.py 控制，改进模式由 my_improve/settings.py 控制；
# 二者必须与训练权重完全一致。
CONFIG_PATH = str(selected_model_config_path())

# 只填写已经通过验证集选定的最终权重。不要根据 test 结果再更换权重。
CHECKPOINT_PATH = Path(r"G:\b1\权重结果\改进实验dfine\dsqc_rba\一般3new2e-4_SD59\best_map50.pth")

TEST_IMAGES_DIR = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\test\images")
TEST_ANNOTATION = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\test\annotations\test.json")
NUM_CLASSES = 1

# 必须与训练和验证时的输入尺寸一致。
INPUT_SIZE = MODEL_IMAGE_SIZE
TEST_BATCH_SIZE = 6
NUM_WORKERS = 2
DEVICE = "cuda"
SEED = 59

# 测试结果使用独立目录，不覆盖训练或验证结果。
TEST_OUTPUT_DIR = Path(r"G:\b1\权重结果\改进实验dfine\dsqc_rba\一般3new2e-4_SD59\测试集精度")

# 只影响论文 TXT 中的显示名称，不参与权重加载。
PAPER_MODEL_NAME = "dsqc_rba_一般3newSD59"

# P/R/F1 固定使用仓库 Validator 的 confidence=0.50、matching IoU=0.50。
# 下列置信度只用于统一效率测试的检测框过滤口径。
FIXED_EVALUATION_CONFIDENCE = 0.50
MATCH_IOU = 0.50
ENABLE_FPS_BENCHMARK = True
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


def replace_resize_size(ops, input_size: int) -> None:
    """使测试集 Resize 与训练输入尺寸保持一致。"""
    for op in ops or []:
        if isinstance(op, dict) and op.get("type") == "Resize":
            op["size"] = [input_size, input_size]


def build_test_config() -> YAMLConfig:
    """读取模型 YAML，并把验证加载器安全地指向独立测试集。"""
    overrides = yaml_utils.load_config(CONFIG_PATH)
    overrides["num_classes"] = NUM_CLASSES
    overrides["remap_mscoco_category"] = False
    overrides["device"] = DEVICE
    overrides["sync_bn"] = False
    overrides["output_dir"] = str(TEST_OUTPUT_DIR)
    overrides["eval_spatial_size"] = [INPUT_SIZE, INPUT_SIZE]
    overrides["resume"] = str(CHECKPOINT_PATH)
    overrides["tuning"] = None

    # D-FINE 没有单独 test_dataloader；独立测试时复用无增强、无 shuffle 的
    # val_dataloader 结构，但将其影像和 JSON 明确替换为 test 集。
    test_loader = overrides["val_dataloader"]
    test_loader["total_batch_size"] = TEST_BATCH_SIZE
    test_loader["num_workers"] = NUM_WORKERS
    test_loader["shuffle"] = False
    test_loader["drop_last"] = False
    test_loader["dataset"]["img_folder"] = TEST_IMAGES_DIR
    test_loader["dataset"]["ann_file"] = TEST_ANNOTATION
    replace_resize_size(test_loader["dataset"]["transforms"].get("ops"), INPUT_SIZE)

    return YAMLConfig(CONFIG_PATH, **overrides)


def validate_paths() -> None:
    """在加载模型前检查测试配置，避免误测验证集或错误权重。"""
    required = {
        "模型配置": Path(CONFIG_PATH),
        "模型权重": CHECKPOINT_PATH,
        "测试影像目录": TEST_IMAGES_DIR,
        "测试标注": TEST_ANNOTATION,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError("以下测试路径不存在：\n" + "\n".join(missing))
    if "test" not in {part.lower() for part in TEST_IMAGES_DIR.parts}:
        raise ValueError(f"TEST_IMAGES_DIR 看起来不是 test 目录：{TEST_IMAGES_DIR}")
    if TEST_ANNOTATION.name.lower() != "test.json":
        raise ValueError(f"测试标注文件必须明确使用 test.json：{TEST_ANNOTATION}")
    if INPUT_SIZE <= 0 or INPUT_SIZE % 32 != 0:
        raise ValueError("INPUT_SIZE 必须为能被 32 整除的正整数。")
    if TEST_BATCH_SIZE <= 0:
        raise ValueError("TEST_BATCH_SIZE 必须大于 0。")
    if FPS_WARMUP_ITERS < 0 or FPS_TEST_ITERS <= 0:
        raise ValueError("FPS_WARMUP_ITERS 必须≥0，FPS_TEST_ITERS 必须>0。")
    if FIXED_EVALUATION_CONFIDENCE != 0.50 or MATCH_IOU != 0.50:
        raise ValueError("论文统一口径要求测试集 P/R/F1 使用 confidence=0.50、IoU=0.50。")
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("DEVICE 设置为 cuda，但当前没有可用 CUDA 显卡。")


def calculate_model_flops(model) -> float:
    """按 D-FINE 官方 calflops 部署图口径统计 GFLOPs。"""
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


def benchmark_single_image_fps(model, postprocessor, data_loader, device: torch.device) -> dict:
    """统计 batch=1 的模型前向、后处理与固定置信度过滤速度。"""
    samples, targets = next(iter(data_loader))
    samples = samples[:1].to(device)
    target = {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in targets[0].items()
    }
    original_sizes = target["orig_size"].reshape(1, 2)

    def forward_and_postprocess() -> None:
        outputs = model(samples)
        results = postprocessor(outputs, original_sizes)
        for result in results:
            keep = result["scores"] >= FIXED_EVALUATION_CONFIDENCE
            result["scores"] = result["scores"][keep]
            result["labels"] = result["labels"][keep]
            result["boxes"] = result["boxes"][keep]

    def synchronize() -> None:
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.inference_mode():
        for _ in range(FPS_WARMUP_ITERS):
            forward_and_postprocess()
        synchronize()
        start = time.perf_counter()
        for _ in range(FPS_TEST_ITERS):
            forward_and_postprocess()
        synchronize()

    seconds_per_image = (time.perf_counter() - start) / FPS_TEST_ITERS
    return {
        "latency_ms_single_image_forward_post": seconds_per_image * 1000.0,
        "FPS_single_image_forward_post": 1.0 / seconds_per_image,
        "warmup_iterations": FPS_WARMUP_ITERS,
        "test_iterations": FPS_TEST_ITERS,
        "efficiency_confidence": FIXED_EVALUATION_CONFIDENCE,
    }


def save_final_comparison_table(
    result_dir: Path,
    coco_metrics: dict,
    threshold_metrics: dict,
    speed_metrics: dict,
    model_params: int,
    flops_g: float,
) -> Path:
    """生成可直接复制到 Word/Excel 的测试集最终核心对比表。"""
    latency = speed_metrics.get("latency_ms_single_image_forward_post")
    fps = speed_metrics.get("FPS_single_image_forward_post")
    flops_text = f"{flops_g:.3f}" if flops_g > 0 else "N/A"
    latency_text = f"{latency:.4f}" if latency is not None else "N/A"
    fps_text = f"{fps:.4f}" if fps is not None else "N/A"

    accuracy_header = (
        f"{'Model':<20}{'Input':>8}{'Params(M)':>14}{'AP50':>12}{'AP75':>12}"
        f"{'mAP50:95':>14}{'P@0.5':>12}{'R@0.5':>12}{'F1@0.5':>12}"
    )
    accuracy_row = (
        f"{PAPER_MODEL_NAME:<20}{INPUT_SIZE:>8}{model_params / 1e6:>14.3f}"
        f"{coco_metrics['AP@0.5']:>12.4f}{coco_metrics['AP@0.75']:>12.4f}"
        f"{coco_metrics['mAP@0.5:0.95']:>14.4f}"
        f"{threshold_metrics['precision']:>12.4f}"
        f"{threshold_metrics['recall']:>12.4f}"
        f"{threshold_metrics['f1']:>12.4f}"
    )
    efficiency_header = (
        f"{'Model':<20}{'Input':>8}{'Params(M)':>14}{'FLOPs(G)':>14}"
        f"{'Latency(ms/image)':>22}{'FPS':>14}"
    )
    efficiency_row = (
        f"{PAPER_MODEL_NAME:<20}{INPUT_SIZE:>8}{model_params / 1e6:>14.3f}"
        f"{flops_text:>14}{latency_text:>22}{fps_text:>14}"
    )

    lines = [
        "D-FINE 独立测试集最终核心对比指标",
        "=" * 112,
        "一、测试集精度对比表",
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
        "1. 本表全部精度来自独立 test 集；test 集未参与训练、选权重或调参。",
        "2. P@0.5、R@0.5、F1@0.5：置信度≥0.50，匹配IoU≥0.50。",
        "3. AP50、AP75、mAP50:95：COCO标准检测评价。",
        f"4. FLOPs：D-FINE官方calflops部署图口径，batch=1，输入{INPUT_SIZE}×{INPUT_SIZE}。",
        "5. Latency与FPS：batch=1，模型前向+检测后处理，不含磁盘和DataLoader。",
        f"6. 测速预热{FPS_WARMUP_ITERS}次，正式测试{FPS_TEST_ITERS}次。",
        "",
        "制表提示：将不同模型生成的数值行汇总，即可形成论文最终核心对比表。",
    ]
    path = result_dir / "测试集最终核心指标.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def save_detailed_report(
    result_dir: Path,
    coco_metrics: dict,
    threshold_metrics: dict,
    speed_metrics: dict,
    model_params: int,
    flops_g: float,
    started_at: datetime,
    elapsed_seconds: float,
) -> Path:
    """保存可追溯但不用于调参的测试集完整指标。"""
    lines = [
        "火电厂目标检测独立测试集精度报告",
        "=" * 52,
        f"模型配置: {CONFIG_PATH}",
        f"模型权重: {CHECKPOINT_PATH}",
        f"测试影像: {TEST_IMAGES_DIR}",
        f"测试标注: {TEST_ANNOTATION}",
        f"输入尺寸: {INPUT_SIZE} × {INPUT_SIZE}",
        "",
        "一、论文核心测试指标",
        "说明：P、R、F1 固定使用置信度0.50、匹配IoU 0.50；不在test集搜索最佳阈值。",
        f"Precision (P)                 : {threshold_metrics['precision']:.4f}",
        f"Recall (R)                    : {threshold_metrics['recall']:.4f}",
        f"F1-score                      : {threshold_metrics['f1']:.4f}",
        f"AP@0.5                        : {coco_metrics['AP@0.5']:.4f}",
        f"AP@0.75                       : {coco_metrics['AP@0.75']:.4f}",
        f"mAP@0.5:0.95                  : {coco_metrics['mAP@0.5:0.95']:.4f}",
        "",
        "二、完整 COCO 检测指标",
    ]
    lines.extend(f"{name:30s}: {value:.4f}" for name, value in coco_metrics.items())
    lines.extend(
        [
            "",
            "三、固定阈值检测统计",
        ]
    )
    for name, value in threshold_metrics.items():
        if name in {"TPs", "FPs", "FNs"}:
            lines.append(f"{name:30s}: {int(value)}")
        else:
            lines.append(f"{name:30s}: {value:.4f}")
    lines.extend(
        [
            "",
            "四、模型规模与效率",
            f"模型参数量 (M)                 : {model_params / 1e6:.3f}",
            f"FLOPs (G)                       : {flops_g:.3f}"
            if flops_g > 0
            else "FLOPs (G)                       : N/A（未安装 calflops）",
        ]
    )
    lines.extend(f"{name:30s}: {value:.4f}" for name, value in speed_metrics.items())
    lines.extend(
        [
            "",
            "五、测试耗时",
            f"开始时间: {started_at.isoformat(timespec='seconds')}",
            f"结束时间: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"总耗时: {format_elapsed_time(elapsed_seconds)}",
            f"总秒数: {elapsed_seconds:.2f}",
        ]
    )
    path = result_dir / "测试集精度报告.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    validate_paths()
    started_at = datetime.now().astimezone()
    start_time = time.perf_counter()

    print("\n========== 火电厂目标检测独立测试 ==========")
    print(f"共享模型选择: {MODEL_TAG}")
    print(f"模型配置: {CONFIG_PATH}")
    print(f"最终权重: {CHECKPOINT_PATH}")
    print(f"测试影像: {TEST_IMAGES_DIR}")
    print(f"测试标注: {TEST_ANNOTATION}")
    print(f"输入尺寸: {INPUT_SIZE} | 测试批量: {TEST_BATCH_SIZE} | 设备: {DEVICE}")
    print(f"结果目录: {TEST_OUTPUT_DIR}\n")

    dist_utils.setup_distributed(print_rank=0, print_method="builtin", seed=SEED)
    try:
        cfg = build_test_config()
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        solver = TASKS[cfg.yaml_cfg["task"]](cfg)
        # Solver.eval() 仅完成模型、EMA、test loader 和 checkpoint 的加载；
        # 正式测试由下面 evaluate() 执行一次。
        solver.eval()
        model = solver.ema.module if solver.ema else solver.model

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
            # 测试集禁止生成/选择最佳置信度，避免用 test 调参。
            return_confidence_metrics=False,
        )

        coco_metrics = dict(zip(COCO_METRIC_NAMES, stats["coco_eval_bbox"]))
        threshold_metrics = {
            name: float(value)
            for name, value in stats.get("detection_metrics", {}).items()
        }
        model_params = sum(parameter.numel() for parameter in model.parameters())
        speed_metrics = (
            benchmark_single_image_fps(
                model, solver.postprocessor, solver.val_dataloader, solver.device
            )
            if ENABLE_FPS_BENCHMARK
            else {}
        )
        elapsed_seconds = time.perf_counter() - start_time

        print("\n========== 独立测试集最终结果 ==========")
        for name in ("AP@0.5", "AP@0.75", "mAP@0.5:0.95"):
            print(f"{name:16s}: {coco_metrics[name]:.4f}")
        for name in ("precision", "recall", "f1", "TPs", "FPs", "FNs"):
            value = threshold_metrics[name]
            print(f"{name:16s}: {int(value) if name in {'TPs', 'FPs', 'FNs'} else f'{value:.4f}'}")
        print(f"模型参数量 (M)  : {model_params / 1e6:.3f}")
        print(f"FLOPs (G)       : {flops_g:.3f}" if flops_g > 0 else "FLOPs (G)       : N/A")
        for name, value in speed_metrics.items():
            print(f"{name:34s}: {value:.4f}")

        TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        metrics_payload = {
            "evaluation_split": "test",
            "test_set_used_for_tuning": False,
            "model_config": CONFIG_PATH,
            "checkpoint": str(CHECKPOINT_PATH),
            "test_images": str(TEST_IMAGES_DIR),
            "test_annotation": str(TEST_ANNOTATION),
            "input_size": INPUT_SIZE,
            "fixed_confidence": FIXED_EVALUATION_CONFIDENCE,
            "matching_iou": MATCH_IOU,
            "coco_metrics": coco_metrics,
            "threshold_metrics": threshold_metrics,
            "speed_metrics": speed_metrics,
            "model_parameters": model_params,
            "FLOPs_G": flops_g if flops_g > 0 else None,
            "test_elapsed_seconds": elapsed_seconds,
        }
        (TEST_OUTPUT_DIR / "test_metrics.json").write_text(
            json.dumps(metrics_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        table_path = save_final_comparison_table(
            TEST_OUTPUT_DIR,
            coco_metrics,
            threshold_metrics,
            speed_metrics,
            model_params,
            flops_g,
        )
        report_path = save_detailed_report(
            TEST_OUTPUT_DIR,
            coco_metrics,
            threshold_metrics,
            speed_metrics,
            model_params,
            flops_g,
            started_at,
            elapsed_seconds,
        )
        torch.save(coco_evaluator.coco_eval["bbox"].eval, TEST_OUTPUT_DIR / "test_eval.pth")

        print(f"\n测试指标 JSON：{TEST_OUTPUT_DIR / 'test_metrics.json'}")
        print(f"测试详细报告：{report_path}")
        print(f"论文核心表格：{table_path}")
        print(f"COCO 评估对象：{TEST_OUTPUT_DIR / 'test_eval.pth'}")
    finally:
        dist_utils.cleanup()


if __name__ == "__main__":
    main()
