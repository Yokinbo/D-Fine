"""
D-FINE 火电厂大范围遥感影像推理策略对比实验。

使用步骤
--------
1. 修改下方“用户配置区”的绝对路径。
2. 将 STRATEGY_MODE 改为 A、B、C、D 或 E。
3. 在 dfine 环境中运行：

       python myscript/大范围遥感影像火电厂推理策略对比实验.py

五组策略
--------
A：512 不重叠窗口 + 全局 NMS
B：512 窗口 / 256 步长 + 全局 NMS
C：B + 边界/不确定候选的 768 扩展视域复检 + 全局 NMS
D：C + 边界风险与跨视图一致性加权融合（BR-DCF）
E：D + 仅对扩展视域窗口执行选择性 TTA

重要说明
--------
* 所谓“边界目标”是被内部 512×512 推理网格截断的目标，不是行政区外边界目标。
* 真值矢量只用于推理结束后的评价，模型推理和复检触发过程不会读取真值。
* 输入源窗口始终是 512×512；模型内部统一缩放至 640×640，与训练配置保持一致。
* mAP 使用全部低阈值候选形成 PR 曲线；Precision/Recall/F1 使用固定置信度阈值。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# PyTorch 2.1 may probe an installed Transformers package while importing ONNX
# helpers. D-FINE inference does not use Hugging Face models, so suppress that
# irrelevant compatibility warning before importing torch.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

# rasterio / geopandas 不属于 D-FINE 原始训练依赖。采用可选导入是为了在缺包时
# 给出明确提示，而不是在脚本第一行直接报难以理解的 ModuleNotFoundError。
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window
except ImportError:
    rasterio = None
    Resampling = None
    Window = None

try:
    import geopandas as gpd
except ImportError:
    gpd = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core import YAMLConfig


# =============================================================================
# 用户配置区：通常只修改这里，然后直接运行脚本
# =============================================================================

# A / B / C / D / E；建议每次只运行一组，输出会进入独立子目录。
STRATEGY_MODE = "A"

# 灵武市或其他镇级/县级 RGB GeoTIFF。影像至少应包含 3 个波段。
INPUT_TIF = r""

# ArcMap 标注的火电厂水平矩形真值框。
# 推荐 Shapefile / GeoPackage；使用这两种格式时需要 geopandas。
# 也支持与影像坐标系完全一致的 GeoJSON（无需 geopandas）。
GT_VECTOR = r""

# D-FINE-S + VRAC 配置及最终训练权重。
MODEL_CONFIG = r"E:\YOLO\D-FINE\configs\dfine\custom\dfine_hgnetv2_s_custom_vrac.yml"
CHECKPOINT = r"E:\YOLO\D-FINE\output\火电厂_D-FINE-S_VRAC_200轮\best_stg1.pth"
# 本任务仅检测火电厂（hdc）。必须与训练权重的检测头类别数一致。
NUM_CLASSES = 1

# 每种策略自动建立 strategy_A、strategy_B ... 子目录。
OUTPUT_ROOT = r"E:\YOLO\D-FINE\output\大范围遥感影像推理策略对比"

DEVICE = "cuda:0"
USE_AMP = True
MODEL_INPUT_SIZE = 640
BATCH_SIZE = 4

# 源影像滑窗参数。A 会自动把步长改为 512，B~E 使用 256。
BASE_TILE_SIZE = 512
OVERLAP_STRIDE = 256

# 低阈值候选用于 COCO 风格 mAP 曲线；最终制图与 P/R/F1 使用 FINAL_CONF。
CANDIDATE_CONF = 0.05
FINAL_CONF = 0.50
GLOBAL_NMS_IOU = 0.50
# 单类别火电厂非常稀疏，每个窗口只保留分数最高的若干低阈值候选，防止全市
# 数千窗口累计数百万个 DETR queries。该限制发生在 mAP 评估前，建议不要低于 20。
MAX_CANDIDATES_PER_WINDOW_VIEW = 50

# C~E：边界风险和不确定候选的扩展视域复检。
EDGE_MARGIN = 64
EDGE_RISK_THRESHOLD = 0.50
REFINE_MIN_CONF = 0.25
REFINE_STABLE_CONF = 0.50
REFINE_UNCERTAIN_CANDIDATES = True
REFINE_CONTEXT_SIZE = 768
REFINE_TRIGGER_NMS_IOU = 0.30
MAX_REFINE_WINDOWS = 1000

# D~E：BR-DCF 融合参数。
FUSION_IOU = 0.35
FUSION_GAMMA = 1.0
FUSION_CENTER_FLOOR = 0.20
FUSION_CONSISTENCY_FLOOR = 0.20

# E：只对复检窗口执行这些 TTA。original 会始终执行，无需写入。
SELECTIVE_TTA_MODES = ("hflip", "vflip", "hvflip")

# 真值评价参数。
MATCH_IOU = 0.50
BOUNDARY_GT_VISIBILITY = 0.90
SEVERE_BOUNDARY_GT_VISIBILITY = 0.70

# 无效黑边处理。有效像素比例过低的窗口不进入模型。
BLACK_THRESHOLD = 3
MIN_VALID_RATIO = 0.20

# 若已在 ArcMap 中得到测试区有效面积，可直接填写；0 表示按有效非黑像素估算。
# 输入影像必须为以米为单位的投影坐标系，才能自动换算 km²。
VALID_AREA_KM2 = 0.0

# 输出选项。GeoJSON 和 CSV 始终可写；GPKG/SHP 需要 geopandas。
WRITE_GPKG = True
WRITE_SHP = False
WRITE_PREVIEW = True
PREVIEW_MAX_SIZE = 2200

# 当前训练数据来自普通 RGB 8-bit 影像。若输入不是 uint8，默认停止，避免静默产生色彩域偏移。
ALLOW_NON_UINT8 = False

# =============================================================================


@dataclass(frozen=True)
class Strategy:
    code: str
    name: str
    stride: int
    use_refine: bool
    use_brdcf: bool
    use_selective_tta: bool


STRATEGIES: Dict[str, Strategy] = {
    "A": Strategy("A", "512不重叠窗口+全局NMS", BASE_TILE_SIZE, False, False, False),
    "B": Strategy("B", "512/256重叠窗口+全局NMS", OVERLAP_STRIDE, False, False, False),
    "C": Strategy("C", "B+边界候选扩展视域复检", OVERLAP_STRIDE, True, False, False),
    "D": Strategy("D", "C+BR-DCF边界一致性加权融合", OVERLAP_STRIDE, True, True, False),
    "E": Strategy("E", "D+选择性TTA", OVERLAP_STRIDE, True, True, True),
}


@dataclass
class WindowSpec:
    x0: int
    y0: int
    size: int
    window_id: str
    is_refine: bool = False


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int = 1
    window_id: str = ""
    view: str = "original"
    is_refine: bool = False
    boundary_risk: float = 0.0
    center_weight: float = 1.0
    support_count: int = 1

    @property
    def box(self) -> np.ndarray:
        return np.asarray([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)

    @property
    def center(self) -> Tuple[float, float]:
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))


@dataclass
class GroundTruth:
    plant_id: str
    x1: float
    y1: float
    x2: float
    y2: float
    max_visible_ratio: float = 1.0
    boundary_level: str = "完整"

    @property
    def box(self) -> np.ndarray:
        return np.asarray([self.x1, self.y1, self.x2, self.y2], dtype=np.float64)


def require_runtime_dependencies() -> None:
    if rasterio is None:
        raise RuntimeError(
            "当前 Python 环境缺少 rasterio，无法读取带地理坐标的 GeoTIFF。\n"
            "请先在专门的大图推理环境中安装 rasterio；若需要直接读取/写出 Shapefile 或 "
            "GeoPackage，还需要 geopandas。脚本不会自动修改你的环境。"
        )


def start_positions(length: int, tile_size: int, stride: int) -> List[int]:
    """Generate full-coverage starts and explicitly include the raster's last edge."""
    if length <= tile_size:
        return [0]
    positions = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.empty((0,), dtype=np.float64)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / np.maximum(area_a + area_b - intersection, 1e-12)


def clip_box(box: Sequence[float], width: int, height: int) -> np.ndarray:
    x1, y1, x2, y2 = [float(value) for value in box]
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    return np.asarray([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])


def local_boundary_properties(local_box: Sequence[float], size: int) -> Tuple[float, float]:
    """Return boundary risk and Hann-like center reliability for a local box."""
    x1, y1, x2, y2 = [float(v) for v in local_box]
    clearance = max(0.0, min(x1, y1, size - x2, size - y2))
    boundary_risk = 1.0 - min(1.0, clearance / max(float(EDGE_MARGIN), 1.0))
    cx = min(max(0.5 * (x1 + x2), 0.0), float(size))
    cy = min(max(0.5 * (y1 + y2), 0.0), float(size))
    hann_center = math.sin(math.pi * cx / size) ** 2 * math.sin(math.pi * cy / size) ** 2
    center_weight = FUSION_CENTER_FLOOR + (1.0 - FUSION_CENTER_FLOOR) * hann_center
    return boundary_risk, center_weight


def apply_tta(batch: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "original":
        return batch
    if mode == "hflip":
        return torch.flip(batch, dims=[3])
    if mode == "vflip":
        return torch.flip(batch, dims=[2])
    if mode == "hvflip":
        return torch.flip(batch, dims=[2, 3])
    raise ValueError(f"不支持的 TTA 模式: {mode}")


def undo_tta_boxes(boxes: np.ndarray, size: int, mode: str) -> np.ndarray:
    boxes = boxes.copy()
    if boxes.size == 0 or mode == "original":
        return boxes
    old = boxes.copy()
    if mode in {"hflip", "hvflip"}:
        boxes[:, 0] = size - old[:, 2]
        boxes[:, 2] = size - old[:, 0]
        old = boxes.copy()
    if mode in {"vflip", "hvflip"}:
        boxes[:, 1] = size - old[:, 3]
        boxes[:, 3] = size - old[:, 1]
    return boxes


class DFineInferenceModel(nn.Module):
    """D-FINE deploy model plus deploy postprocessor."""

    def __init__(self, config_path: str, checkpoint_path: str, device: torch.device):
        super().__init__()
        # 原始 D-FINE YAML 默认面向 COCO（777 类）。训练与 valid.py 都会在
        # 创建模型前覆盖 num_classes；大图推理也必须进行同样覆盖，否则单类别
        # 权重无法加载到 777 类检测头中。
        cfg = YAMLConfig(
            config_path,
            num_classes=NUM_CLASSES,
            remap_mscoco_category=False,
        )
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        elif "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint

        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy().to(device).eval()
        self.postprocessor = cfg.postprocessor.deploy().to(device).eval()

    def forward(self, images: torch.Tensor, original_sizes: torch.Tensor):
        outputs = self.model(images)
        return self.postprocessor(outputs, original_sizes)


def load_model(device: torch.device) -> DFineInferenceModel:
    if not Path(MODEL_CONFIG).is_file():
        raise FileNotFoundError(f"模型配置不存在: {MODEL_CONFIG}")
    if not Path(CHECKPOINT).is_file():
        raise FileNotFoundError(f"模型权重不存在: {CHECKPOINT}")
    model = DFineInferenceModel(MODEL_CONFIG, CHECKPOINT, device).to(device).eval()
    dummy = torch.zeros((1, 3, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE), device=device)
    size = torch.tensor([[BASE_TILE_SIZE, BASE_TILE_SIZE]], device=device)
    with torch.inference_mode():
        for _ in range(2):
            with torch.autocast(device_type=device.type, enabled=USE_AMP and device.type == "cuda"):
                model(dummy, size)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return model


def convert_patch_to_uint8(patch: np.ndarray) -> np.ndarray:
    if patch.dtype == np.uint8:
        return patch
    if not ALLOW_NON_UINT8:
        raise TypeError(
            f"输入影像类型为 {patch.dtype}，但模型训练影像按 uint8 RGB 读取。"
            "请先确认影像辐射范围，或明确设置 ALLOW_NON_UINT8=True。"
        )
    info = np.iinfo(patch.dtype) if np.issubdtype(patch.dtype, np.integer) else None
    maximum = float(info.max) if info else float(np.nanmax(patch))
    return np.clip(patch.astype(np.float32) / max(maximum, 1.0) * 255.0, 0, 255).astype(np.uint8)


def read_rgb_patch(src: Any, spec: WindowSpec) -> Tuple[np.ndarray, float]:
    window = Window(spec.x0, spec.y0, spec.size, spec.size)
    patch = src.read([1, 2, 3], window=window, boundless=True, fill_value=0)
    patch = convert_patch_to_uint8(patch)
    valid = ~np.all(patch <= BLACK_THRESHOLD, axis=0)
    try:
        mask = src.dataset_mask(window=window, boundless=True) > 0
        valid &= mask
    except Exception:
        pass
    return patch, float(valid.mean())


def patches_to_tensor(patches: Sequence[np.ndarray], device: torch.device) -> torch.Tensor:
    arrays = np.stack(patches, axis=0)
    tensor = torch.from_numpy(arrays).to(device=device, dtype=torch.float32) / 255.0
    if tensor.shape[-2:] != (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE):
        tensor = F.interpolate(
            tensor,
            size=(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE),
            mode="bilinear",
            align_corners=False,
        )
    return tensor


def decode_model_output(output: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(output, (tuple, list)) and len(output) == 3:
        return output[0], output[1], output[2]
    if isinstance(output, dict):
        return output["labels"], output["boxes"], output["scores"]
    raise TypeError(f"无法识别 D-FINE 输出类型: {type(output)}")


def infer_batch(
    model: DFineInferenceModel,
    patches: Sequence[np.ndarray],
    specs: Sequence[WindowSpec],
    device: torch.device,
    views: Sequence[str],
    raster_width: int,
    raster_height: int,
) -> List[Detection]:
    batch = patches_to_tensor(patches, device)
    all_detections: List[Detection] = []

    for view in views:
        augmented = apply_tta(batch, view)
        original_sizes = torch.tensor(
            [[spec.size, spec.size] for spec in specs], dtype=torch.int64, device=device
        )
        with torch.inference_mode(), torch.autocast(
            device_type=device.type, enabled=USE_AMP and device.type == "cuda"
        ):
            output = model(augmented, original_sizes)
        labels, boxes, scores = decode_model_output(output)

        labels_np = labels.detach().cpu().numpy()
        boxes_np = boxes.detach().cpu().numpy()
        scores_np = scores.detach().cpu().numpy()

        for batch_index, spec in enumerate(specs):
            local_boxes = undo_tta_boxes(boxes_np[batch_index], spec.size, view)
            valid_indices = np.flatnonzero(scores_np[batch_index] >= CANDIDATE_CONF)
            if valid_indices.size > MAX_CANDIDATES_PER_WINDOW_VIEW:
                order = np.argsort(scores_np[batch_index][valid_indices])[::-1]
                valid_indices = valid_indices[order[:MAX_CANDIDATES_PER_WINDOW_VIEW]]
            for prediction_index in valid_indices:
                label = labels_np[batch_index][prediction_index]
                local_box = local_boxes[prediction_index]
                score = scores_np[batch_index][prediction_index]
                boundary_risk, center_weight = local_boundary_properties(local_box, spec.size)
                global_box = clip_box(
                    [
                        local_box[0] + spec.x0,
                        local_box[1] + spec.y0,
                        local_box[2] + spec.x0,
                        local_box[3] + spec.y0,
                    ],
                    raster_width,
                    raster_height,
                )
                if global_box[2] - global_box[0] < 1 or global_box[3] - global_box[1] < 1:
                    continue
                all_detections.append(
                    Detection(
                        *global_box.tolist(),
                        score=float(score),
                        class_id=int(label) + 1,
                        window_id=spec.window_id,
                        view=view,
                        is_refine=spec.is_refine,
                        boundary_risk=float(boundary_risk),
                        center_weight=float(center_weight),
                    )
                )
    return all_detections


def flush_window_batch(
    model: DFineInferenceModel,
    patches: List[np.ndarray],
    specs: List[WindowSpec],
    device: torch.device,
    views: Sequence[str],
    raster_width: int,
    raster_height: int,
) -> List[Detection]:
    if not patches:
        return []
    detections = infer_batch(
        model, patches, specs, device, views, raster_width, raster_height
    )
    patches.clear()
    specs.clear()
    return detections


def format_duration(seconds: float) -> str:
    """Format elapsed/remaining time for the Windows terminal progress bar."""
    seconds = max(0, int(round(seconds)))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def print_window_progress(
    label: str,
    completed: int,
    total: int,
    start_time: float,
    detail: str = "",
) -> None:
    """Print one in-place progress bar with elapsed time and estimated remaining time."""
    completed = min(max(completed, 0), total)
    elapsed = max(time.perf_counter() - start_time, 1e-6)
    ratio = completed / total if total else 1.0
    eta = elapsed * (total - completed) / max(completed, 1)
    bar_width = 24
    filled = int(round(bar_width * ratio))
    bar = "#" * filled + "-" * (bar_width - filled)
    suffix = f" | {detail}" if detail else ""
    print(
        f"\r{label} [{bar}] {ratio:6.2%} | {completed}/{total} | "
        f"耗时 {format_duration(elapsed)} | 预计剩余 {format_duration(eta)}{suffix}",
        end="",
        flush=True,
    )


def run_base_windows(
    src: Any,
    model: DFineInferenceModel,
    strategy: Strategy,
    device: torch.device,
) -> Tuple[List[Detection], int, int]:
    xs = start_positions(src.width, BASE_TILE_SIZE, strategy.stride)
    ys = start_positions(src.height, BASE_TILE_SIZE, strategy.stride)
    total = len(xs) * len(ys)
    used = 0
    skipped = 0
    detections: List[Detection] = []
    patches: List[np.ndarray] = []
    specs: List[WindowSpec] = []
    start_time = time.perf_counter()
    # 约每 1% 刷新一次；小影像至少每个窗口刷新，避免看起来像“卡住”。
    progress_interval = max(1, total // 100)

    print(f"[基础滑窗] 总窗口: {total} | stride={strategy.stride}")
    for row, y0 in enumerate(ys):
        for col, x0 in enumerate(xs):
            spec = WindowSpec(x0, y0, BASE_TILE_SIZE, f"base_r{row}_c{col}")
            patch, valid_ratio = read_rgb_patch(src, spec)
            if valid_ratio < MIN_VALID_RATIO:
                skipped += 1
                continue
            patches.append(patch)
            specs.append(spec)
            used += 1
            if len(patches) >= BATCH_SIZE:
                detections.extend(
                    flush_window_batch(
                        model,
                        patches,
                        specs,
                        device,
                        ("original",),
                        src.width,
                        src.height,
                    )
                )
            completed = row * len(xs) + col + 1
            if completed % progress_interval == 0 or completed == total:
                print_window_progress(
                    "[基础滑窗]",
                    completed,
                    total,
                    start_time,
                    f"有效 {used} | 跳过 {skipped} | 候选框 {len(detections)}",
                )
    detections.extend(
        flush_window_batch(
            model,
            patches,
            specs,
            device,
            ("original",),
            src.width,
            src.height,
        )
    )
    # 最后一个未满批次完成后，再刷新一次，确保进度条和候选框数是最终值。
    print_window_progress(
        "[基础滑窗]",
        total,
        total,
        start_time,
        f"有效 {used} | 跳过 {skipped} | 候选框 {len(detections)}",
    )
    print(f"\n[基础滑窗] 完成 | 有效 {used} | 跳过 {skipped} | 候选框 {len(detections)}")
    return detections, used, skipped


def global_nms(detections: Sequence[Detection], iou_threshold: float) -> List[Detection]:
    if not detections:
        return []
    boxes = torch.tensor(np.stack([det.box for det in detections]), dtype=torch.float32)
    scores = torch.tensor([det.score for det in detections], dtype=torch.float32)
    keep = nms(boxes, scores, iou_threshold).cpu().tolist()
    return [detections[index] for index in keep]


def select_refine_windows(
    base_detections: Sequence[Detection], width: int, height: int
) -> List[WindowSpec]:
    candidates = [
        det
        for det in base_detections
        if det.score >= REFINE_MIN_CONF
        and (
            det.boundary_risk >= EDGE_RISK_THRESHOLD
            or (REFINE_UNCERTAIN_CANDIDATES and det.score < REFINE_STABLE_CONF)
        )
    ]
    # 相邻重叠切片可能同时产生同一候选。先以较低 IoU 做一次去重，防止重复复检。
    candidates = global_nms(candidates, REFINE_TRIGGER_NMS_IOU)
    candidates.sort(
        key=lambda det: det.score * (0.5 + 0.5 * det.boundary_risk), reverse=True
    )
    candidates = candidates[:MAX_REFINE_WINDOWS]

    half = REFINE_CONTEXT_SIZE // 2
    windows: List[WindowSpec] = []
    used_centers: List[Tuple[int, int]] = []
    minimum_center_distance = max(32, BASE_TILE_SIZE // 4)
    for index, det in enumerate(candidates):
        cx, cy = det.center
        center = (int(round(cx)), int(round(cy)))
        if any(
            math.hypot(center[0] - old[0], center[1] - old[1]) < minimum_center_distance
            for old in used_centers
        ):
            continue
        used_centers.append(center)
        windows.append(
            WindowSpec(
                x0=center[0] - half,
                y0=center[1] - half,
                size=REFINE_CONTEXT_SIZE,
                window_id=f"refine_{index:05d}",
                is_refine=True,
            )
        )
    return windows


def run_refine_windows(
    src: Any,
    model: DFineInferenceModel,
    windows: Sequence[WindowSpec],
    strategy: Strategy,
    device: torch.device,
) -> List[Detection]:
    if not windows:
        print("[扩展复检] 没有候选窗口。")
        return []
    views = ("original",) + SELECTIVE_TTA_MODES if strategy.use_selective_tta else ("original",)
    detections: List[Detection] = []
    patches: List[np.ndarray] = []
    specs: List[WindowSpec] = []
    start_time = time.perf_counter()
    total = len(windows)
    progress_interval = max(1, total // 100)
    print(f"[扩展复检] 窗口: {len(windows)} | context={REFINE_CONTEXT_SIZE} | views={views}")
    for index, spec in enumerate(windows):
        patch, valid_ratio = read_rgb_patch(src, spec)
        if valid_ratio < MIN_VALID_RATIO:
            continue
        patches.append(patch)
        specs.append(spec)
        if len(patches) >= BATCH_SIZE:
            detections.extend(
                flush_window_batch(
                    model, patches, specs, device, views, src.width, src.height
                )
            )
        completed = index + 1
        if completed % progress_interval == 0 or completed == total:
            print_window_progress(
                "[扩展复检]",
                completed,
                total,
                start_time,
                f"候选框 {len(detections)} | 视图数 {len(views)}",
            )
    detections.extend(
        flush_window_batch(model, patches, specs, device, views, src.width, src.height)
    )
    print_window_progress(
        "[扩展复检]",
        total,
        total,
        start_time,
        f"候选框 {len(detections)} | 视图数 {len(views)}",
    )
    print(f"\n[扩展复检] 完成 | 候选框 {len(detections)}")
    return detections


def cluster_detections(detections: Sequence[Detection], threshold: float) -> List[List[int]]:
    """Spatial-hash + union-find clustering for large-image candidate boxes.

    Boxes with positive IoU must share at least one spatial bucket, so there is
    no need for an O(N²) all-pairs comparison across the entire city image.
    """
    count = len(detections)
    if count == 0:
        return []
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    boxes = np.stack([det.box for det in detections])
    bucket_size = float(BASE_TILE_SIZE)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for index, box in enumerate(boxes):
        min_col = math.floor(box[0] / bucket_size)
        max_col = math.floor(max(box[0], box[2] - 1e-6) / bucket_size)
        min_row = math.floor(box[1] / bucket_size)
        max_row = math.floor(max(box[1], box[3] - 1e-6) / bucket_size)
        keys = [
            (row, col)
            for row in range(min_row, max_row + 1)
            for col in range(min_col, max_col + 1)
        ]
        possible = sorted({old for key in keys for old in buckets.get(key, [])})
        if possible:
            overlaps = box_iou_one_to_many(box, boxes[np.asarray(possible, dtype=int)])
            for local_index in np.flatnonzero(overlaps >= threshold):
                union(index, possible[int(local_index)])
        for key in keys:
            buckets.setdefault(key, []).append(index)

    groups: Dict[int, List[int]] = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def brdcf_fusion(detections: Sequence[Detection]) -> List[Detection]:
    """Boundary-risk and cross-view consistency weighted box fusion."""
    if not detections:
        return []
    clusters = cluster_detections(detections, FUSION_IOU)
    fused: List[Detection] = []
    for cluster in clusters:
        members = [detections[index] for index in cluster]
        boxes = np.stack([det.box for det in members])
        if len(members) == 1:
            consistency = np.ones(1, dtype=np.float64)
        else:
            consistency_values = []
            for index, box in enumerate(boxes):
                others = np.delete(boxes, index, axis=0)
                consistency_values.append(float(box_iou_one_to_many(box, others).mean()))
            consistency = np.asarray(consistency_values)

        scores = np.asarray([det.score for det in members], dtype=np.float64)
        centers = np.asarray([det.center_weight for det in members], dtype=np.float64)
        consistency_weight = FUSION_CONSISTENCY_FLOOR + (
            1.0 - FUSION_CONSISTENCY_FLOOR
        ) * consistency
        weights = np.power(scores, FUSION_GAMMA) * centers * consistency_weight
        weights = np.maximum(weights, 1e-12)
        fused_box = np.sum(boxes * weights[:, None], axis=0) / weights.sum()
        fused_score = float(np.sum(scores * weights) / weights.sum())
        best = members[int(np.argmax(scores))]
        fused.append(
            Detection(
                *fused_box.tolist(),
                score=fused_score,
                class_id=best.class_id,
                window_id="BRDCF",
                view="fused",
                is_refine=any(det.is_refine for det in members),
                boundary_risk=max(det.boundary_risk for det in members),
                center_weight=float(np.max(centers)),
                support_count=len({(det.window_id, det.view) for det in members}),
            )
        )
    fused.sort(key=lambda det: det.score, reverse=True)
    return fused


def flatten_coordinates(value: Any) -> Iterable[Tuple[float, float]]:
    if (
        isinstance(value, (list, tuple))
        and len(value) >= 2
        and isinstance(value[0], (int, float))
        and isinstance(value[1], (int, float))
    ):
        yield float(value[0]), float(value[1])
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_coordinates(item)


def world_bounds_to_pixel_box(bounds: Sequence[float], transform: Any) -> np.ndarray:
    min_x, min_y, max_x, max_y = [float(v) for v in bounds]
    inverse = ~transform
    pixels = [
        inverse * (min_x, min_y),
        inverse * (min_x, max_y),
        inverse * (max_x, min_y),
        inverse * (max_x, max_y),
    ]
    xs = [point[0] for point in pixels]
    ys = [point[1] for point in pixels]
    return np.asarray([min(xs), min(ys), max(xs), max(ys)], dtype=np.float64)


def load_ground_truths(path: str, src: Any) -> List[GroundTruth]:
    if not path:
        return []
    gt_path = Path(path)
    if not gt_path.is_file():
        raise FileNotFoundError(f"真值矢量不存在: {gt_path}")

    records: List[Tuple[str, Sequence[float]]] = []
    suffix = gt_path.suffix.lower()
    if suffix in {".json", ".geojson"} and gpd is None:
        payload = json.loads(gt_path.read_text(encoding="utf-8"))
        for index, feature in enumerate(payload.get("features", []), start=1):
            coordinates = list(flatten_coordinates(feature.get("geometry", {}).get("coordinates", [])))
            if not coordinates:
                continue
            xs, ys = zip(*coordinates)
            properties = feature.get("properties", {})
            plant_id = str(properties.get("plant_id", properties.get("id", f"GT{index:03d}")))
            records.append((plant_id, (min(xs), min(ys), max(xs), max(ys))))
        print("[提示] 未安装 geopandas：按 GeoJSON 与影像坐标系完全一致处理，不执行重投影。")
    else:
        if gpd is None:
            raise RuntimeError(
                "读取 Shapefile/GeoPackage 需要 geopandas。也可以先在 ArcMap 中导出为与影像"
                "坐标系一致的 GeoJSON。"
            )
        frame = gpd.read_file(gt_path)
        if frame.crs is None:
            raise ValueError("真值矢量缺少坐标系定义。")
        if src.crs is not None and frame.crs != src.crs:
            frame = frame.to_crs(src.crs)
        for index, row in frame.iterrows():
            geometry = row.geometry
            if geometry is None or geometry.is_empty:
                continue
            plant_id = str(row.get("plant_id", row.get("id", f"GT{index + 1:03d}")))
            records.append((plant_id, geometry.bounds))

    ground_truths = []
    for plant_id, bounds in records:
        box = clip_box(world_bounds_to_pixel_box(bounds, src.transform), src.width, src.height)
        if box[2] > box[0] and box[3] > box[1]:
            ground_truths.append(GroundTruth(plant_id, *box.tolist()))
    return ground_truths


def classify_boundary_ground_truths(
    ground_truths: Sequence[GroundTruth], width: int, height: int
) -> None:
    """Classify GT against the fixed A-group non-overlap grid for all strategies."""
    xs = start_positions(width, BASE_TILE_SIZE, BASE_TILE_SIZE)
    ys = start_positions(height, BASE_TILE_SIZE, BASE_TILE_SIZE)
    for gt in ground_truths:
        area = max((gt.x2 - gt.x1) * (gt.y2 - gt.y1), 1e-12)
        best = 0.0
        for y0 in ys:
            if y0 >= gt.y2 or y0 + BASE_TILE_SIZE <= gt.y1:
                continue
            for x0 in xs:
                if x0 >= gt.x2 or x0 + BASE_TILE_SIZE <= gt.x1:
                    continue
                intersection_w = max(0.0, min(gt.x2, x0 + BASE_TILE_SIZE) - max(gt.x1, x0))
                intersection_h = max(0.0, min(gt.y2, y0 + BASE_TILE_SIZE) - max(gt.y1, y0))
                best = max(best, intersection_w * intersection_h / area)
        gt.max_visible_ratio = float(best)
        if best < SEVERE_BOUNDARY_GT_VISIBILITY:
            gt.boundary_level = "严重截断"
        elif best < BOUNDARY_GT_VISIBILITY:
            gt.boundary_level = "中度截断"
        elif best < 1.0 - 1e-9:
            gt.boundary_level = "轻微跨界"
        else:
            gt.boundary_level = "完整"


def greedy_match(
    detections: Sequence[Detection],
    ground_truths: Sequence[GroundTruth],
    iou_threshold: float,
    confidence_threshold: float,
) -> Tuple[int, int, int, Dict[int, int]]:
    selected = [det for det in detections if det.score >= confidence_threshold]
    selected.sort(key=lambda det: det.score, reverse=True)
    gt_boxes = np.stack([gt.box for gt in ground_truths]) if ground_truths else np.empty((0, 4))
    unmatched = set(range(len(ground_truths)))
    matched_gt_to_prediction: Dict[int, int] = {}
    tp = 0
    fp = 0
    for prediction_index, det in enumerate(selected):
        if not unmatched:
            fp += 1
            continue
        indices = np.asarray(sorted(unmatched), dtype=int)
        overlaps = box_iou_one_to_many(det.box, gt_boxes[indices])
        best_local = int(np.argmax(overlaps))
        if overlaps[best_local] >= iou_threshold:
            gt_index = int(indices[best_local])
            unmatched.remove(gt_index)
            matched_gt_to_prediction[gt_index] = prediction_index
            tp += 1
        else:
            fp += 1
    return tp, fp, len(unmatched), matched_gt_to_prediction


def average_precision(
    detections: Sequence[Detection], ground_truths: Sequence[GroundTruth], iou_threshold: float
) -> float:
    if not ground_truths:
        return float("nan")
    selected = sorted(detections, key=lambda det: det.score, reverse=True)
    gt_boxes = np.stack([gt.box for gt in ground_truths])
    unmatched = set(range(len(ground_truths)))
    true_positive = []
    false_positive = []
    for det in selected:
        if unmatched:
            indices = np.asarray(sorted(unmatched), dtype=int)
            overlaps = box_iou_one_to_many(det.box, gt_boxes[indices])
            best_local = int(np.argmax(overlaps))
            if overlaps[best_local] >= iou_threshold:
                unmatched.remove(int(indices[best_local]))
                true_positive.append(1.0)
                false_positive.append(0.0)
                continue
        true_positive.append(0.0)
        false_positive.append(1.0)
    if not true_positive:
        return 0.0
    tp_cumulative = np.cumsum(true_positive)
    fp_cumulative = np.cumsum(false_positive)
    recall = tp_cumulative / len(ground_truths)
    precision = tp_cumulative / np.maximum(tp_cumulative + fp_cumulative, 1e-12)
    # COCO 风格的 101 点插值 AP。
    interpolated = []
    for recall_level in np.linspace(0.0, 1.0, 101):
        valid = precision[recall >= recall_level]
        interpolated.append(float(valid.max()) if valid.size else 0.0)
    return float(np.mean(interpolated))


def count_duplicate_boxes(
    detections: Sequence[Detection], ground_truths: Sequence[GroundTruth]
) -> int:
    selected = [det for det in detections if det.score >= FINAL_CONF]
    if not selected or not ground_truths:
        return 0
    boxes = np.stack([det.box for det in selected])
    duplicate_count = 0
    for gt in ground_truths:
        matches = int(np.sum(box_iou_one_to_many(gt.box, boxes) >= MATCH_IOU))
        duplicate_count += max(0, matches - 1)
    return duplicate_count


def compute_metrics(
    final_detections: Sequence[Detection],
    raw_detections: Sequence[Detection],
    ground_truths: Sequence[GroundTruth],
    inference_seconds: float,
    valid_area_km2: float,
    base_windows: int,
    refine_windows: int,
) -> Dict[str, Any]:
    thresholds = np.arange(0.50, 0.951, 0.05)
    ap_values = [average_precision(final_detections, ground_truths, float(t)) for t in thresholds]
    tp, fp, fn, matches = greedy_match(
        final_detections, ground_truths, MATCH_IOU, FINAL_CONF
    )
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    boundary_indices = {
        index
        for index, gt in enumerate(ground_truths)
        if gt.max_visible_ratio < BOUNDARY_GT_VISIBILITY
    }
    severe_indices = {
        index
        for index, gt in enumerate(ground_truths)
        if gt.max_visible_ratio < SEVERE_BOUNDARY_GT_VISIBILITY
    }
    boundary_tp = len(boundary_indices.intersection(matches.keys()))
    severe_tp = len(severe_indices.intersection(matches.keys()))

    return {
        "AP@0.5": ap_values[0],
        "mAP@0.5:0.95": float(np.mean(ap_values)),
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "boundary_gt_count": len(boundary_indices),
        "boundary_recall": boundary_tp / len(boundary_indices) if boundary_indices else None,
        "severe_boundary_gt_count": len(severe_indices),
        "severe_boundary_recall": severe_tp / len(severe_indices) if severe_indices else None,
        "raw_duplicate_boxes": count_duplicate_boxes(raw_detections, ground_truths),
        "final_duplicate_boxes": count_duplicate_boxes(final_detections, ground_truths),
        "inference_seconds": inference_seconds,
        "valid_area_km2": valid_area_km2,
        "seconds_per_km2": inference_seconds / valid_area_km2 if valid_area_km2 > 0 else None,
        "base_windows": base_windows,
        "refine_windows": refine_windows,
        "refine_window_ratio": refine_windows / base_windows if base_windows else 0.0,
    }


def estimate_valid_area_km2(src: Any) -> float:
    if VALID_AREA_KM2 > 0:
        return float(VALID_AREA_KM2)
    if src.crs is None or not getattr(src.crs, "is_projected", False):
        raise ValueError(
            "自动计算 km² 需要以米为单位的投影坐标系。当前影像不是投影坐标系，"
            "请在 ArcMap 中计算有效面积并填写 VALID_AREA_KM2。"
        )
    valid_pixels = 0
    for _, window in src.block_windows(1):
        rgb = src.read([1, 2, 3], window=window)
        valid = ~np.all(rgb <= BLACK_THRESHOLD, axis=0)
        try:
            valid &= src.dataset_mask(window=window) > 0
        except Exception:
            pass
        valid_pixels += int(valid.sum())
    transform = src.transform
    pixel_area = abs(transform.a * transform.e - transform.b * transform.d)
    return valid_pixels * pixel_area / 1_000_000.0


def pixel_box_to_world_polygon(box: Sequence[float], transform: Any) -> List[List[float]]:
    x1, y1, x2, y2 = [float(v) for v in box]
    pixels = [(x1, y1), (x2, y1), (x2, y2), (x1, y2), (x1, y1)]
    return [[float(x), float(y)] for x, y in (transform * point for point in pixels)]


def save_predictions_csv(
    path: Path, detections: Sequence[Detection], transform: Any
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "prediction_id",
                "class_name",
                "score",
                "pixel_x1",
                "pixel_y1",
                "pixel_x2",
                "pixel_y2",
                "center_geo_x",
                "center_geo_y",
                "support_count",
                "contains_refine_view",
            ]
        )
        for index, det in enumerate(detections, start=1):
            cx, cy = det.center
            geo_x, geo_y = transform * (cx, cy)
            writer.writerow(
                [
                    f"P{index:04d}",
                    "火电厂",
                    f"{det.score:.6f}",
                    f"{det.x1:.3f}",
                    f"{det.y1:.3f}",
                    f"{det.x2:.3f}",
                    f"{det.y2:.3f}",
                    f"{geo_x:.6f}",
                    f"{geo_y:.6f}",
                    det.support_count,
                    int(det.is_refine),
                ]
            )


def save_predictions_geojson(
    path: Path, detections: Sequence[Detection], transform: Any, crs: Any
) -> None:
    features = []
    for index, det in enumerate(detections, start=1):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "prediction_id": f"P{index:04d}",
                    "class_name": "火电厂",
                    "score": det.score,
                    "support_count": det.support_count,
                    "refined": bool(det.is_refine),
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [pixel_box_to_world_polygon(det.box, transform)],
                },
            }
        )
    payload: Dict[str, Any] = {"type": "FeatureCollection", "features": features}
    if crs is not None:
        payload["crs"] = {"type": "name", "properties": {"name": crs.to_string()}}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_optional_geopandas_vectors(
    result_dir: Path, detections: Sequence[Detection], transform: Any, crs: Any
) -> None:
    if not (WRITE_GPKG or WRITE_SHP):
        return
    if gpd is None:
        print("[提示] 未安装 geopandas，已跳过 GPKG/SHP；GeoJSON 和 CSV 已正常输出。")
        return
    try:
        from shapely.geometry import Polygon

        records = []
        for index, det in enumerate(detections, start=1):
            records.append(
                {
                    "pred_id": f"P{index:04d}",
                    "class": "hdc",
                    "score": det.score,
                    "support": det.support_count,
                    "refined": int(det.is_refine),
                    "geometry": Polygon(pixel_box_to_world_polygon(det.box, transform)),
                }
            )
        frame = gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
        if WRITE_GPKG:
            frame.to_file(result_dir / "predictions.gpkg", layer="power_plants", driver="GPKG")
        if WRITE_SHP:
            frame.to_file(result_dir / "predictions.shp", encoding="utf-8")
    except Exception as error:
        print(f"[提示] GPKG/SHP 输出失败，但不影响其他结果: {error}")


def save_preview(
    src: Any, path: Path, detections: Sequence[Detection], ground_truths: Sequence[GroundTruth]
) -> None:
    scale = min(1.0, PREVIEW_MAX_SIZE / max(src.width, src.height))
    preview_width = max(1, int(round(src.width * scale)))
    preview_height = max(1, int(round(src.height * scale)))
    rgb = src.read(
        [1, 2, 3],
        out_shape=(3, preview_height, preview_width),
        resampling=Resampling.bilinear,
    )
    rgb = convert_patch_to_uint8(rgb)
    image = Image.fromarray(np.transpose(rgb, (1, 2, 0)), mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for gt in ground_truths:
        box = tuple(float(v) * scale for v in gt.box)
        draw.rectangle(box, outline=(0, 255, 0), width=2)
        draw.text((box[0], box[1]), gt.plant_id, fill=(0, 255, 0), font=font)
    for index, det in enumerate(detections, start=1):
        box = tuple(float(v) * scale for v in det.box)
        draw.rectangle(box, outline=(255, 0, 0), width=2)
        draw.text((box[0], max(0, box[1] - 10)), f"P{index}:{det.score:.2f}", fill=(255, 0, 0), font=font)
    image.save(path)


def save_boundary_gt_csv(path: Path, ground_truths: Sequence[GroundTruth]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["plant_id", "pixel_x1", "pixel_y1", "pixel_x2", "pixel_y2", "max_visible_ratio", "boundary_level"]
        )
        for gt in ground_truths:
            writer.writerow(
                [
                    gt.plant_id,
                    f"{gt.x1:.3f}",
                    f"{gt.y1:.3f}",
                    f"{gt.x2:.3f}",
                    f"{gt.y2:.3f}",
                    f"{gt.max_visible_ratio:.6f}",
                    gt.boundary_level,
                ]
            )


def save_metrics_txt(path: Path, strategy: Strategy, metrics: Dict[str, Any]) -> None:
    def fmt(value: Any) -> str:
        if value is None:
            return "N/A（该子集没有真值目标）"
        if isinstance(value, float):
            return f"{value:.6f}"
        return str(value)

    lines = [
        "大范围遥感影像火电厂推理策略评价报告",
        "=" * 56,
        f"策略: {strategy.code} - {strategy.name}",
        f"输入影像: {INPUT_TIF}",
        f"模型权重: {CHECKPOINT}",
        "",
    ]
    lines.extend(f"{key}: {fmt(value)}" for key, value in metrics.items())
    lines.extend(
        [
            "",
            "口径说明：",
            f"1. P/R/F1：confidence >= {FINAL_CONF}，IoU >= {MATCH_IOU}。",
            "2. mAP：使用 0.50:0.05:0.95 的 101 点插值 AP。",
            f"3. 边界目标：相对固定 A 组网格的最大可见比例 < {BOUNDARY_GT_VISIBILITY}。",
            "4. 推理时间不含模型加载、面积估算和输出文件绘制，包含窗口读取、模型前向、复检与融合。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def update_comparison_summary(path: Path, strategy: Strategy, metrics: Dict[str, Any]) -> None:
    fieldnames = [
        "strategy",
        "strategy_name",
        "AP@0.5",
        "mAP@0.5:0.95",
        "Precision",
        "Recall",
        "F1",
        "boundary_recall",
        "severe_boundary_recall",
        "raw_duplicate_boxes",
        "final_duplicate_boxes",
        "seconds_per_km2",
        "refine_window_ratio",
        "base_windows",
        "refine_windows",
    ]
    existing: Dict[str, Dict[str, Any]] = {}
    if path.is_file():
        with path.open("r", newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                existing[row["strategy"]] = row
    row = {"strategy": strategy.code, "strategy_name": strategy.name}
    row.update({name: metrics.get(name) for name in fieldnames if name not in row})
    existing[strategy.code] = row
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for code in "ABCDE":
            if code in existing:
                writer.writerow(existing[code])


def resolve_device() -> torch.device:
    if DEVICE.startswith("cuda") and not torch.cuda.is_available():
        print("[提示] CUDA 不可用，自动改用 CPU。")
        return torch.device("cpu")
    return torch.device(DEVICE)


def validate_config(strategy_code: str) -> Strategy:
    require_runtime_dependencies()
    code = strategy_code.upper().strip()
    if code not in STRATEGIES:
        raise ValueError(f"STRATEGY_MODE 必须是 A/B/C/D/E，当前为 {strategy_code!r}")
    if not INPUT_TIF:
        raise ValueError("请先在用户配置区填写 INPUT_TIF。")
    if not Path(INPUT_TIF).is_file():
        raise FileNotFoundError(f"输入 GeoTIFF 不存在: {INPUT_TIF}")
    if BASE_TILE_SIZE <= 0 or OVERLAP_STRIDE <= 0:
        raise ValueError("窗口大小和步长必须大于 0。")
    if OVERLAP_STRIDE > BASE_TILE_SIZE:
        raise ValueError("OVERLAP_STRIDE 不能大于 BASE_TILE_SIZE。")
    if not 0 <= CANDIDATE_CONF <= FINAL_CONF <= 1:
        raise ValueError("置信度应满足 0 <= CANDIDATE_CONF <= FINAL_CONF <= 1。")
    return STRATEGIES[code]


def run_experiment(strategy_code: str) -> None:
    strategy = validate_config(strategy_code)
    result_dir = Path(OUTPUT_ROOT) / f"strategy_{strategy.code}"
    result_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device()

    print("\n" + "=" * 68)
    print(f"推理策略: {strategy.code} - {strategy.name}")
    print(f"输入影像: {INPUT_TIF}")
    print(f"真值矢量: {GT_VECTOR or '未设置（只推理，不计算精度）'}")
    print(f"模型权重: {CHECKPOINT}")
    print(f"设备: {device} | AMP: {USE_AMP}")
    print(f"输出目录: {result_dir}")
    print("=" * 68)

    model = load_model(device)
    with rasterio.open(INPUT_TIF) as src:
        if src.count < 3:
            raise ValueError("输入影像至少需要 3 个 RGB 波段。")
        ground_truths = load_ground_truths(GT_VECTOR, src)
        classify_boundary_ground_truths(ground_truths, src.width, src.height)
        save_boundary_gt_csv(result_dir / "boundary_ground_truths.csv", ground_truths)
        valid_area_km2 = estimate_valid_area_km2(src)
        print(f"[数据] 影像尺寸: {src.width} × {src.height} | CRS: {src.crs}")
        print(f"[数据] 真值目标: {len(ground_truths)} | 有效面积: {valid_area_km2:.3f} km²")
        print(
            f"[数据] 边界目标: {sum(gt.max_visible_ratio < BOUNDARY_GT_VISIBILITY for gt in ground_truths)} | "
            f"严重截断: {sum(gt.max_visible_ratio < SEVERE_BOUNDARY_GT_VISIBILITY for gt in ground_truths)}"
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_start = time.perf_counter()

        base_detections, base_windows, skipped_windows = run_base_windows(
            src, model, strategy, device
        )
        refine_windows_specs: List[WindowSpec] = []
        refine_detections: List[Detection] = []
        if strategy.use_refine:
            refine_windows_specs = select_refine_windows(
                base_detections, src.width, src.height
            )
            refine_detections = run_refine_windows(
                src, model, refine_windows_specs, strategy, device
            )

        raw_detections = base_detections + refine_detections
        if strategy.use_brdcf:
            final_detections = brdcf_fusion(raw_detections)
        else:
            final_detections = global_nms(raw_detections, GLOBAL_NMS_IOU)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds = time.perf_counter() - inference_start

        # mAP 使用所有低阈值融合结果；制图仅输出最终阈值以上目标。
        export_detections = [det for det in final_detections if det.score >= FINAL_CONF]
        metrics: Dict[str, Any] = {
            "inference_seconds": inference_seconds,
            "valid_area_km2": valid_area_km2,
            "seconds_per_km2": inference_seconds / valid_area_km2,
            "base_windows": base_windows,
            "skipped_windows": skipped_windows,
            "refine_windows": len(refine_windows_specs),
            "refine_window_ratio": len(refine_windows_specs) / base_windows if base_windows else 0.0,
            "raw_candidate_boxes": len(raw_detections),
            "final_candidate_boxes": len(final_detections),
            "exported_boxes": len(export_detections),
        }
        if ground_truths:
            metrics = compute_metrics(
                final_detections,
                raw_detections,
                ground_truths,
                inference_seconds,
                valid_area_km2,
                base_windows,
                len(refine_windows_specs),
            ) | {
                "skipped_windows": skipped_windows,
                "raw_candidate_boxes": len(raw_detections),
                "final_candidate_boxes": len(final_detections),
                "exported_boxes": len(export_detections),
            }

        save_predictions_csv(result_dir / "predictions.csv", export_detections, src.transform)
        save_predictions_geojson(
            result_dir / "predictions.geojson", export_detections, src.transform, src.crs
        )
        save_optional_geopandas_vectors(
            result_dir, export_detections, src.transform, src.crs
        )
        if WRITE_PREVIEW:
            save_preview(
                src,
                result_dir / "prediction_preview.png",
                export_detections,
                ground_truths,
            )

    metadata = {
        "strategy": asdict(strategy),
        "input_tif": INPUT_TIF,
        "gt_vector": GT_VECTOR,
        "model_config": MODEL_CONFIG,
        "checkpoint": CHECKPOINT,
        "device": str(device),
        "model_input_size": MODEL_INPUT_SIZE,
        "base_tile_size": BASE_TILE_SIZE,
        "candidate_conf": CANDIDATE_CONF,
        "final_conf": FINAL_CONF,
        "global_nms_iou": GLOBAL_NMS_IOU,
        "edge_margin": EDGE_MARGIN,
        "edge_risk_threshold": EDGE_RISK_THRESHOLD,
        "refine_context_size": REFINE_CONTEXT_SIZE,
        "selective_tta_modes": list(SELECTIVE_TTA_MODES),
        "fusion_iou": FUSION_IOU,
        "metrics": metrics,
    }
    (result_dir / "run_config_and_metrics.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    save_metrics_txt(result_dir / "metrics.txt", strategy, metrics)
    update_comparison_summary(
        Path(OUTPUT_ROOT) / "strategy_comparison_summary.csv", strategy, metrics
    )

    print("\n========== 本次实验结果 ==========")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:28s}: {value:.6f}")
        else:
            print(f"{key:28s}: {value}")
    print(f"结果已保存: {result_dir}")
    print(f"五组汇总表: {Path(OUTPUT_ROOT) / 'strategy_comparison_summary.csv'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="D-FINE 大范围遥感影像火电厂推理策略对比")
    parser.add_argument(
        "--strategy",
        choices=list("ABCDE"),
        default=STRATEGY_MODE,
        help="覆盖脚本顶部的 STRATEGY_MODE",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args().strategy)
