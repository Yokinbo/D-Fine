"""
D-FINE 火电厂大范围遥感影像正式应用脚本。

固定使用对比实验中的 E 策略：

    512/256 重叠滑窗
    + 边界/不确定候选 768 扩展视域复检
    + BR-DCF 跨窗口一致性加权融合
    + 仅对复检窗口执行选择性 TTA

本脚本不需要真值标注，适合直接对灵武市、镇级或县级 RGB GeoTIFF 进行应用检测。
算法实现复用“ 大范围遥感影像火电厂推理策略对比实验.py ”中的核心函数，保证论文
对比实验和正式应用使用完全相同的窗口、复检、TTA 与融合逻辑。

使用方法：
1. 修改下方“用户配置区”的 INPUT_TIF、CHECKPOINT 和 OUTPUT_DIR。
2. 在 dfine 环境中运行：

       python myscript/大范围遥感影像推理正式实用版.py

也可以临时从命令行覆盖路径：

       python myscript/大范围遥感影像推理正式实用版.py --input "D:\灵武市.tif"
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Dict

# Running a script inside myscript/ puts that directory first on sys.path.
# Add the repository root so this script reads the same network input size as
# train.py and valid.py.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import (
    MODEL_IMAGE_SIZE,
    MODEL_TAG,
    selected_model_config_path,
)


# =============================================================================
# 用户配置区：通常只修改这几项
# =============================================================================

# 待检测的灵武市/镇级/县级 RGB GeoTIFF 绝对路径。
INPUT_TIF = r"G:\金三角tif影像\宁夏\银川市\灵武市1.88m\Level16\灵武市1.88m.tif"

# 由 experiment_config.py 的 MODEL_SIZE 自动选择 S/M 结构。
MODEL_CONFIG = str(selected_model_config_path())

# 训练集仅含火电厂（hdc）一个类别；必须与所选 checkpoint 的检测头一致。
NUM_CLASSES = 1

# 建议填写经过 valid.py 比较后确定的最佳正式训练权重。
CHECKPOINT = r"G:\b\8月13模型权重结果\six_m_512_200epoch\best_map50.pth"

# 本次正式应用的独立输出目录。
OUTPUT_DIR = r"G:\b\8月13精度对比测试可删\大图推理效果\m-512_200"

# Shapefile 输出文件名：只填写文件名并保留 .shp 后缀，文件仍保存到 OUTPUT_DIR。
SHP_OUTPUT_NAME = "m-512_200灵武市-置信度0.5.shp"

# 最终制图置信度：先填写 valid.py 报告的最佳 F1 置信度，再根据
# 真实大图上的误检/漏检人工调整。也可用 --confidence 临时覆盖。
FINAL_CONFIDENCE = 0.5


DEVICE = "cuda:0"
USE_AMP = True
BATCH_SIZE = 6



# 跨重叠窗口/复检视图的支持次数。1 表示关闭此过滤；应用场景可尝试改为 4。
# 支持次数越高，候选框越稳定，但过高可能漏掉边缘或只被少数窗口覆盖的目标。
#最终只保留至少被 4 个不同窗口或 TTA 视图共同支持的检测框。意思是一个目标
# 得有4个框同时认定是目标，最终才会保留这个目标的框，建议尝试1、2、4。越大越严格。

MIN_SUPPORT_COUNT = 1

# E 策略核心参数。默认值应与对比实验保持一致，论文定稿后不要随意修改。
BASE_TILE_SIZE = 512                       #可以做两组实验：BASE_TILE_SIZE = 768   基础窗口：768×768；
OVERLAP_STRIDE = 256                                     #OVERLAP_STRIDE = 384   相邻窗口步长：384；
REFINE_CONTEXT_SIZE = 768                                 #REFINE_CONTEXT_SIZE = 1024   复检窗口：1024×1024；
EDGE_MARGIN = 64                                    #实验1：512窗口，步长256，复检768
EDGE_RISK_THRESHOLD = 0.50                          #实验2：768窗口，步长384，复检1024
REFINE_MIN_CONFIDENCE = 0.25
REFINE_STABLE_CONFIDENCE = 0.50
FUSION_IOU = 0.35

# 输出 GeoJSON 和 CSV 不需要 geopandas；GPKG/SHP 需要 geopandas、shapely、fiona。
WRITE_GPKG = True
# 需要 Shapefile 时保持 True；首次运行前请安装对应地理空间依赖。
WRITE_SHP = True
WRITE_PREVIEW = True
PREVIEW_MAX_SIZE = 2400

# =============================================================================


CORE_SCRIPT_NAME = "大范围遥感影像火电厂推理策略对比实验.py"


def load_inference_core() -> ModuleType:
    """Load the tested comparison implementation without duplicating its algorithms."""
    core_path = Path(__file__).resolve().with_name(CORE_SCRIPT_NAME)
    if not core_path.is_file():
        raise FileNotFoundError(f"找不到核心推理脚本: {core_path}")
    module_name = "dfine_large_raster_inference_core"
    spec = importlib.util.spec_from_file_location(module_name, core_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载核心推理脚本: {core_path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass 在执行模块时需要能够通过 sys.modules 找到所属模块。
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def apply_application_config(core: ModuleType, input_tif: str, checkpoint: str) -> None:
    """Synchronize the formal application settings into the shared inference core."""
    core.INPUT_TIF = input_tif
    core.GT_VECTOR = ""
    core.MODEL_CONFIG = MODEL_CONFIG
    core.CHECKPOINT = checkpoint
    core.NUM_CLASSES = NUM_CLASSES
    core.DEVICE = DEVICE
    core.USE_AMP = USE_AMP
    core.BATCH_SIZE = BATCH_SIZE
    core.MODEL_INPUT_SIZE = MODEL_IMAGE_SIZE
    core.FINAL_CONF = FINAL_CONFIDENCE
    core.BASE_TILE_SIZE = BASE_TILE_SIZE
    core.OVERLAP_STRIDE = OVERLAP_STRIDE
    core.REFINE_CONTEXT_SIZE = REFINE_CONTEXT_SIZE
    core.EDGE_MARGIN = EDGE_MARGIN
    core.EDGE_RISK_THRESHOLD = EDGE_RISK_THRESHOLD
    core.REFINE_MIN_CONF = REFINE_MIN_CONFIDENCE
    core.REFINE_STABLE_CONF = REFINE_STABLE_CONFIDENCE
    core.FUSION_IOU = FUSION_IOU
    core.WRITE_GPKG = WRITE_GPKG
    core.WRITE_SHP = WRITE_SHP
    core.SHP_OUTPUT_NAME = SHP_OUTPUT_NAME
    core.WRITE_PREVIEW = WRITE_PREVIEW
    core.PREVIEW_MAX_SIZE = PREVIEW_MAX_SIZE

    # Strategy 对象在核心模块导入时已建立。这里显式重建 E，确保顶部参数修改后同步生效。
    core.STRATEGIES["E"] = core.Strategy(
        code="E",
        name="512/256重叠滑窗+768复检+BR-DCF+选择性TTA",
        stride=OVERLAP_STRIDE,
        use_refine=True,
        use_brdcf=True,
        use_selective_tta=True,
    )


def validate_paths(core: ModuleType, input_tif: str, checkpoint: str) -> None:
    core.require_runtime_dependencies()
    required = {
        "输入 GeoTIFF": input_tif,
        "模型配置": MODEL_CONFIG,
        "模型权重": checkpoint,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not path or not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("以下路径未正确填写：\n" + "\n".join(missing))
    if BASE_TILE_SIZE not in {512, 768, 1024}:
        raise ValueError("BASE_TILE_SIZE 建议使用 512、768 或 1024。")
    if not 0 < OVERLAP_STRIDE <= BASE_TILE_SIZE:
        raise ValueError("OVERLAP_STRIDE 必须满足 0 < stride <= BASE_TILE_SIZE。")
    if REFINE_CONTEXT_SIZE < BASE_TILE_SIZE:
        raise ValueError("REFINE_CONTEXT_SIZE 不能小于基础窗口。")
    if MODEL_IMAGE_SIZE <= 0 or MODEL_IMAGE_SIZE % 32 != 0:
        raise ValueError("MODEL_IMAGE_SIZE 必须是能被 32 整除的正整数。")
    if not 0.0 <= FINAL_CONFIDENCE <= 1.0:
        raise ValueError("FINAL_CONFIDENCE 必须在 [0, 1] 范围内。")
    if Path(SHP_OUTPUT_NAME).name != SHP_OUTPUT_NAME or Path(SHP_OUTPUT_NAME).suffix.lower() != ".shp":
        raise ValueError("SHP_OUTPUT_NAME 只能填写文件名，且必须保留 .shp 后缀。")


def synchronize_cuda(core: ModuleType, device: Any) -> None:
    if device.type == "cuda":
        core.torch.cuda.synchronize(device)


def save_application_summary(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        f"{MODEL_TAG} 火电厂大范围遥感影像正式应用报告",
        "=" * 58,
        "推理策略: E（重叠滑窗 + 扩展视域复检 + BR-DCF + 选择性TTA）",
        f"输入影像: {summary['input_tif']}",
        f"模型权重: {summary['checkpoint']}",
        f"影像尺寸: {summary['raster_width']} × {summary['raster_height']}",
        f"坐标系: {summary['crs']}",
        "",
        f"有效基础窗口数: {summary['base_windows']}",
        f"跳过无效窗口数: {summary['skipped_windows']}",
        f"二次复检窗口数: {summary['refine_windows']}",
        f"二次复检窗口占比: {summary['refine_window_ratio']:.4%}",
        f"基础候选框数: {summary['base_candidate_boxes']}",
        f"复检/TTA候选框数: {summary['refine_candidate_boxes']}",
        f"BR-DCF融合后低阈值候选数: {summary['fused_candidate_boxes']}",
        f"最终输出火电厂数量(conf≥{FINAL_CONFIDENCE:.2f}): {summary['exported_boxes']}",
        f"推理总耗时: {summary['inference_seconds']:.2f} 秒",
        "",
        "说明：耗时包含窗口读取、模型前向、选择性复检/TTA和BR-DCF融合，",
        "不包含模型加载、GeoJSON/GPKG写出及预览图绘制。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_formal_inference(input_tif: str, checkpoint: str, output_dir: str) -> None:
    core = load_inference_core()
    apply_application_config(core, input_tif, checkpoint)
    validate_paths(core, input_tif, checkpoint)

    result_dir = Path(output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    device = core.resolve_device()
    strategy = core.STRATEGIES["E"]

    print("\n" + "=" * 72)
    print(f"{MODEL_TAG} 火电厂大范围遥感影像正式推理")
    print(f"共享模型选择: {MODEL_TAG}")
    print(f"策略: E - {strategy.name}")
    print(f"输入: {input_tif}")
    print(f"权重: {checkpoint}")
    print(f"输出: {result_dir}")
    print(f"设备: {device} | AMP: {USE_AMP} | batch: {BATCH_SIZE}")
    print(f"网络输入尺寸: {MODEL_IMAGE_SIZE}×{MODEL_IMAGE_SIZE} | 最终置信度: {FINAL_CONFIDENCE:.3f}")
    print("=" * 72)

    # 模型加载与预热不计入正式大图推理时间。
    print("[模型] 正在加载单类别火电厂权重并进行 GPU 预热，请稍候...")
    model = core.load_model(device)
    print("[模型] 权重加载和预热完成。")

    print("[影像] 正在打开 GeoTIFF 并读取空间参考...")
    with core.rasterio.open(input_tif) as src:
        if src.count < 3:
            raise ValueError("输入影像至少需要 3 个 RGB 波段。")
        print(f"[影像] size={src.width}×{src.height} | bands={src.count} | dtype={src.dtypes[0]}")
        print(f"[影像] CRS={src.crs} | transform={src.transform}")

        synchronize_cuda(core, device)
        start_time = time.perf_counter()

        base_detections, base_windows, skipped_windows = core.run_base_windows(
            src, model, strategy, device
        )
        refine_specs = core.select_refine_windows(base_detections, src.width, src.height)
        refine_detections = core.run_refine_windows(
            src, model, refine_specs, strategy, device
        )
        raw_detection_count = len(base_detections) + len(refine_detections)
        print(f"[BR-DCF] 正在融合 {raw_detection_count} 个跨窗口候选框...")
        fused_detections = core.brdcf_fusion(base_detections + refine_detections)
        print(f"[BR-DCF] 融合完成，剩余 {len(fused_detections)} 个候选框。")

        synchronize_cuda(core, device)
        inference_seconds = time.perf_counter() - start_time

        final_detections = [
            detection
            for detection in fused_detections
            if detection.score >= FINAL_CONFIDENCE
            and detection.support_count >= MIN_SUPPORT_COUNT
        ]
        final_detections.sort(key=lambda detection: detection.score, reverse=True)

        core.save_predictions_csv(
            result_dir / "火电厂检测结果.csv", final_detections, src.transform
        )
        core.save_predictions_geojson(
            result_dir / "火电厂检测结果.geojson",
            final_detections,
            src.transform,
            src.crs,
        )
        # 复用核心写出函数时，它会使用固定英文文件名 predictions.gpkg/shp。
        core.save_optional_geopandas_vectors(
            result_dir, final_detections, src.transform, src.crs
        )
        if WRITE_PREVIEW:
            core.save_preview(
                src,
                result_dir / "火电厂检测预览图.png",
                final_detections,
                [],
            )

        summary: Dict[str, Any] = {
            "strategy": "E",
            "input_tif": input_tif,
            "checkpoint": checkpoint,
            "model_config": MODEL_CONFIG,
            "model_image_size": MODEL_IMAGE_SIZE,
            "output_dir": str(result_dir),
            "device": str(device),
            "crs": str(src.crs),
            "raster_width": src.width,
            "raster_height": src.height,
            "base_tile_size": BASE_TILE_SIZE,
            "base_stride": OVERLAP_STRIDE,
            "refine_context_size": REFINE_CONTEXT_SIZE,
            "tta_modes": ["original", *core.SELECTIVE_TTA_MODES],
            "final_confidence": FINAL_CONFIDENCE,
            "min_support_count": MIN_SUPPORT_COUNT,
            "base_windows": base_windows,
            "skipped_windows": skipped_windows,
            "refine_windows": len(refine_specs),
            "refine_window_ratio": len(refine_specs) / base_windows if base_windows else 0.0,
            "base_candidate_boxes": len(base_detections),
            "refine_candidate_boxes": len(refine_detections),
            "fused_candidate_boxes": len(fused_detections),
            "exported_boxes": len(final_detections),
            "inference_seconds": inference_seconds,
        }

    (result_dir / "正式推理运行参数.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_application_summary(result_dir / "正式推理结果摘要.txt", summary)

    print("\n========== 正式推理完成 ==========")
    print(f"最终检测数量: {summary['exported_boxes']}")
    print(f"二次复检窗口: {summary['refine_windows']} ({summary['refine_window_ratio']:.2%})")
    print(f"推理耗时: {summary['inference_seconds']:.2f} 秒")
    print(f"GeoJSON: {result_dir / '火电厂检测结果.geojson'}")
    print(f"CSV: {result_dir / '火电厂检测结果.csv'}")
    if WRITE_PREVIEW:
        print(f"预览图: {result_dir / '火电厂检测预览图.png'}")
    print(f"运行摘要: {result_dir / '正式推理结果摘要.txt'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{MODEL_TAG} 火电厂大范围遥感影像 E 策略正式推理")
    parser.add_argument("--input", default=INPUT_TIF, help="覆盖顶部 INPUT_TIF")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="覆盖顶部 CHECKPOINT")
    parser.add_argument("--output", default=OUTPUT_DIR, help="覆盖顶部 OUTPUT_DIR")
    parser.add_argument(
        "--confidence",
        type=float,
        default=FINAL_CONFIDENCE,
        help="覆盖顶部 FINAL_CONFIDENCE，便于人工比较不同推理阈值",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    FINAL_CONFIDENCE = arguments.confidence
    run_formal_inference(arguments.input, arguments.checkpoint, arguments.output)
