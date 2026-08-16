"""逐张预测小幅 RGB TIFF，并输出检测结果与漏检/误检清单。

本脚本用于已经切好的验证/测试影像，不负责把一张大幅 GeoTIFF 再切窗。它与
YOLO26 的同名脚本使用相同的输入、标签和统计口径，便于对比困难样本。

在仓库根目录、D-FINE 环境中执行：

    python myscript/切片推理预测.py

运行前请先修改下方“用户配置区”的路径和阈值。
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

# 避免 PyTorch 间接导入 Transformers 时输出与 D-FINE 无关的兼容性警告。
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torchvision.ops import nms

try:
    import rasterio
except ImportError:
    rasterio = None


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_config import MODEL_IMAGE_SIZE, selected_model_config_path
from src.core import YAMLConfig


# =============================================================================
# 用户配置区
# =============================================================================

# 模型结构必须与训练权重一致。默认跟随 experiment_config.py 中的 S/M 设置。
MODEL_CONFIG = selected_model_config_path()
CHECKPOINT_PATH = Path(r"E:\YOLO\D-FINE\output\新版数据集m-512_100轮\best_map50.pth")
NUM_CLASSES = 1

# 待预测的小 TIFF 目录，支持递归读取子目录。
INPUT_DIR = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\test\images")

# 对应的 YOLO 标签目录；只推理、不统计漏检误检时设为 None。
LABEL_DIR= Path(r"E:\YOLO\D-FINE\datasets\mydatasets\test\labels")
#LABEL_DIR = None

OUTPUT_DIR = Path(r"F:\2testkeshan\可删模型切片推测\3")

CONFIDENCE = 0.50
NMS_IOU = 0.70
MATCH_IOU = 0.50  # 仅用于预测框与 YOLO 真值框匹配，统计 TP/FP/FN。
DEVICE = "cuda:0"
USE_AMP = True
BATCH_SIZE = 6
MAX_DETECTIONS = 300
CLASS_NAME = "hdc"

# =============================================================================

TIFF_SUFFIXES = {".tif", ".tiff"}


class DFineInferenceModel(nn.Module):
    """D-FINE 部署模型及其检测后处理器。"""

    def __init__(self, config_path: Path, checkpoint_path: Path, device: torch.device):
        super().__init__()
        cfg = YAMLConfig(
            str(config_path),
            num_classes=NUM_CLASSES,
            remap_mscoco_category=False,
            eval_spatial_size=[MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE],
        )
        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if isinstance(checkpoint, dict) and "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        elif isinstance(checkpoint, dict) and "model" in checkpoint:
            state = checkpoint["model"]
        else:
            state = checkpoint

        cfg.model.load_state_dict(state)
        self.model = cfg.model.deploy().to(device).eval()
        self.postprocessor = cfg.postprocessor.deploy().to(device).eval()

    def forward(self, images: torch.Tensor, original_sizes: torch.Tensor):
        return self.postprocessor(self.model(images), original_sizes)


def validate_config() -> torch.device:
    """在加载模型前检查路径、数值和运行设备。"""
    if rasterio is None:
        raise RuntimeError("当前环境缺少 rasterio，无法读取 TIFF；请先安装 rasterio。")
    required = {"模型配置": Path(MODEL_CONFIG), "模型权重": CHECKPOINT_PATH}
    missing = [f"{name}：{path}" for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError("以下文件不存在：\n" + "\n".join(missing))
    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(f"待预测目录不存在：{INPUT_DIR}")
    if LABEL_DIR is not None and not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"标签目录不存在：{LABEL_DIR}")
    if MODEL_IMAGE_SIZE <= 0 or MODEL_IMAGE_SIZE % 32:
        raise ValueError("MODEL_IMAGE_SIZE 必须为能被 32 整除的正整数。")
    if not all(0 <= value <= 1 for value in (CONFIDENCE, NMS_IOU, MATCH_IOU)):
        raise ValueError("CONFIDENCE、NMS_IOU 和 MATCH_IOU 必须位于 [0, 1]。")
    if BATCH_SIZE <= 0 or MAX_DETECTIONS <= 0:
        raise ValueError("BATCH_SIZE 和 MAX_DETECTIONS 必须大于 0。")

    requested = torch.device(DEVICE)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[提示] CUDA 不可用，自动改用 CPU。")
        return torch.device("cpu")
    return requested


def find_tiffs(directory: Path) -> list[Path]:
    """按固定顺序递归查找 TIFF。"""
    return sorted(
        path for path in directory.rglob("*")
        if path.is_file() and path.suffix.lower() in TIFF_SUFFIXES
    )


def read_rgb_tif(path: Path) -> np.ndarray:
    """读取前三个波段，返回 uint8 RGB HWC 数组。"""
    with rasterio.open(path) as source:
        if source.count < 3:
            raise ValueError(f"TIFF 少于 3 个波段，无法作为 RGB 输入：{path}")
        dtypes = source.dtypes[:3]
        if any(dtype != "uint8" for dtype in dtypes):
            raise ValueError(
                f"TIFF 前三个波段不是 uint8，不能保证与训练数据一致：{path} ({dtypes})"
            )
        return np.moveaxis(source.read([1, 2, 3]), 0, -1)


def images_to_tensor(images: list[np.ndarray], device: torch.device) -> torch.Tensor:
    """将不同原始尺寸的 RGB 图像缩放并组成模型输入批次。"""
    tensors = []
    for image in images:
        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        tensors.append(tensor.float().div_(255.0))
    batch = torch.stack(
        [
            F.interpolate(
                tensor.unsqueeze(0),
                size=(MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            for tensor in tensors
        ]
    )
    return batch.to(device, non_blocking=device.type == "cuda")


def decode_model_output(output: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """兼容 D-FINE 部署后处理器的元组和字典输出。"""
    if isinstance(output, (tuple, list)) and len(output) == 3:
        return output[0], output[1], output[2]
    if isinstance(output, dict):
        return output["labels"], output["boxes"], output["scores"]
    raise TypeError(f"无法识别 D-FINE 输出类型：{type(output)}")


def postprocess_one(
    labels: torch.Tensor, boxes: torch.Tensor, scores: torch.Tensor
) -> np.ndarray:
    """执行置信度过滤、逐类别 NMS 和最大检测数限制。"""
    keep = scores >= CONFIDENCE
    labels, boxes, scores = labels[keep], boxes[keep], scores[keep]
    if not scores.numel():
        return np.empty((0, 6), dtype=np.float32)

    kept_indices = []
    for class_id in labels.unique():
        class_indices = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        selected = nms(boxes[class_indices], scores[class_indices], NMS_IOU)
        kept_indices.append(class_indices[selected])
    keep = torch.cat(kept_indices)
    keep = keep[torch.argsort(scores[keep], descending=True)[:MAX_DETECTIONS]]

    return np.column_stack(
        (
            labels[keep].detach().cpu().numpy(),
            boxes[keep].detach().cpu().numpy(),
            scores[keep].detach().cpu().numpy(),
        )
    ).astype(np.float32, copy=False)


def predict_batch(
    model: DFineInferenceModel, images: list[np.ndarray], device: torch.device
) -> list[np.ndarray]:
    """预测一个图像批次，返回每张图的 [class, x1, y1, x2, y2, score]。"""
    batch = images_to_tensor(images, device)
    original_sizes = torch.tensor(
        [[image.shape[1], image.shape[0]] for image in images],
        dtype=torch.int64,
        device=device,
    )
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        enabled=USE_AMP and device.type == "cuda",
    ):
        output = model(batch, original_sizes)
    labels, boxes, scores = decode_model_output(output)
    return [
        postprocess_one(labels[index], boxes[index], scores[index])
        for index in range(len(images))
    ]


def read_labels(path: Path | None, width: int, height: int) -> np.ndarray:
    """读取 YOLO 归一化标签，返回 [class, x1, y1, x2, y2]。"""
    if path is None or not path.is_file():
        return np.empty((0, 5), dtype=np.float32)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return np.empty((0, 5), dtype=np.float32)

    targets = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        values = line.split()
        if len(values) != 5:
            raise ValueError(f"标签格式错误：{path} 第 {line_number} 行")
        class_id, center_x, center_y, box_width, box_height = map(float, values)
        targets.append(
            (
                int(class_id),
                (center_x - box_width / 2) * width,
                (center_y - box_height / 2) * height,
                (center_x + box_width / 2) * width,
                (center_y + box_height / 2) * height,
            )
        )
    return np.asarray(targets, dtype=np.float32)


def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """计算一个 xyxy 框与多个 xyxy 框的 IoU。"""
    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    intersection = np.maximum(0, intersection_x2 - intersection_x1) * np.maximum(
        0, intersection_y2 - intersection_y1
    )
    box_area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    boxes_area = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    return intersection / (box_area + boxes_area - intersection + 1e-9)


def match_predictions(predictions: np.ndarray, targets: np.ndarray) -> tuple[int, int, int]:
    """按置信度顺序，将预测框贪心匹配到同类别真值框。"""
    if not len(predictions):
        return 0, 0, len(targets)
    if not len(targets):
        return 0, len(predictions), 0

    matched_targets: set[int] = set()
    true_positives = 0
    for prediction in predictions[np.argsort(-predictions[:, 5])]:
        candidates = np.where(targets[:, 0] == prediction[0])[0]
        candidates = np.asarray(
            [index for index in candidates if int(index) not in matched_targets], dtype=int
        )
        if not len(candidates):
            continue
        overlaps = box_iou(prediction[1:5], targets[candidates, 1:5])
        best_target = int(candidates[int(overlaps.argmax())])
        if float(overlaps.max()) >= MATCH_IOU:
            matched_targets.add(best_target)
            true_positives += 1
    return true_positives, len(predictions) - true_positives, len(targets) - true_positives


def save_preview(
    path: Path,
    rgb: np.ndarray,
    predictions: np.ndarray,
    targets: np.ndarray,
) -> None:
    """保存同时包含真值框（绿色）与预测框（红色）的预览 PNG。"""
    image = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    line_width = max(2, round(max(image.size) / 320))

    ground_truth_color = (0, 220, 80)
    prediction_color = (255, 40, 40)

    def draw_text_label(x: float, y: float, text: str, color: tuple[int, int, int]) -> None:
        """绘制带深色底的标签，避免文字与遥感影像背景混在一起。"""
        x = max(0.0, min(float(x), image.width - 1.0))
        y = max(0.0, min(float(y), image.height - 1.0))
        left, top, right, bottom = draw.textbbox((x, y), text, font=font)
        padding = 2
        background = (
            max(0, left - padding),
            max(0, top - padding),
            min(image.width - 1, right + padding),
            min(image.height - 1, bottom + padding),
        )
        draw.rectangle(background, fill=(0, 0, 0))
        draw.text((x, y), text, fill=color, font=font)

    # 真值框先画成稍粗的绿色框；当预测框与真值框高度重合时，外侧仍能看到绿色。
    for class_id, x1, y1, x2, y2 in targets:
        box = (float(x1), float(y1), float(x2), float(y2))
        draw.rectangle(box, outline=ground_truth_color, width=line_width + 2)
        class_text = CLASS_NAME if int(class_id) == 0 else str(int(class_id))
        draw_text_label(float(x1), max(0.0, float(y1) - 14), f"GT {class_text}", ground_truth_color)

    # 预测框使用红色，并在标签中显示置信度。
    for class_id, x1, y1, x2, y2, confidence in predictions:
        box = (float(x1), float(y1), float(x2), float(y2))
        draw.rectangle(box, outline=prediction_color, width=line_width)
        class_text = CLASS_NAME if int(class_id) == 0 else str(int(class_id))
        draw_text_label(
            float(x1),
            min(float(image.height - 12), max(0.0, float(y1) + 2)),
            f"PRED {class_text} {confidence:.2f}",
            prediction_color,
        )

    # 固定图例：即使某张图没有真值框或预测框，也能明确辨认颜色含义。
    legend = "GT: GREEN   PRED: RED"
    legend_box = draw.textbbox((6, 6), legend, font=font)
    draw.rectangle(
        (2, 2, legend_box[2] + 10, legend_box[3] + 10),
        fill=(0, 0, 0),
        outline=(255, 255, 255),
        width=1,
    )
    draw.text((6, 6), "GT: GREEN", fill=ground_truth_color, font=font)
    gt_width = draw.textlength("GT: GREEN   ", font=font)
    draw.text((6 + gt_width, 6), "PRED: RED", fill=prediction_color, font=font)

    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def save_yolo_predictions(
    path: Path, predictions: np.ndarray, width: int, height: int
) -> None:
    """按 YOLO 的 class cx cy w h confidence 格式保存预测标签。"""
    lines = []
    for class_id, x1, y1, x2, y2, confidence in predictions:
        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        lines.append(
            f"{int(class_id)} {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f} {confidence:.6f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def batched(items: list[Path], size: int):
    """按固定大小切分列表，兼容 Python 3.9+。"""
    for start in range(0, len(items), size):
        yield items[start : start + size]


def main() -> None:
    """逐批推理 TIFF，保存预测结果并生成误差分析报告。"""
    device = validate_config()
    image_paths = find_tiffs(INPUT_DIR)
    if not image_paths:
        raise FileNotFoundError(f"未在目录中找到 TIFF：{INPUT_DIR}")

    output_images = OUTPUT_DIR / "预测预览图"
    output_labels = OUTPUT_DIR / "预测标签"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"发现 {len(image_paths)} 张 TIFF，正在加载模型：{CHECKPOINT_PATH.name}")
    print(f"配置：{MODEL_CONFIG} | 输入尺寸：{MODEL_IMAGE_SIZE} | 设备：{device}")
    model = DFineInferenceModel(Path(MODEL_CONFIG), CHECKPOINT_PATH, device)

    per_image_rows = []
    detection_rows = []
    no_prediction_images = []
    missed_target_images = []
    false_positive_images = []
    total_tp = total_fp = total_fn = 0
    completed = 0

    for batch_paths in batched(image_paths, BATCH_SIZE):
        images = [read_rgb_tif(path) for path in batch_paths]
        batch_predictions = predict_batch(model, images, device)

        for image_path, rgb, predictions in zip(batch_paths, images, batch_predictions):
            relative_path = image_path.relative_to(INPUT_DIR)
            label_path = LABEL_DIR / relative_path.with_suffix(".txt") if LABEL_DIR else None
            targets = read_labels(label_path, rgb.shape[1], rgb.shape[0])
            tp, fp, fn = match_predictions(predictions, targets)
            total_tp += tp
            total_fp += fp
            total_fn += fn

            save_preview(
                (output_images / relative_path).with_suffix(".png"),
                rgb,
                predictions,
                targets,
            )
            save_yolo_predictions(
                (output_labels / relative_path).with_suffix(".txt"),
                predictions,
                rgb.shape[1],
                rgb.shape[0],
            )
            if not len(predictions):
                no_prediction_images.append(str(relative_path))
            if fn:
                missed_target_images.append(str(relative_path))
            if fp:
                false_positive_images.append(str(relative_path))

            per_image_rows.append(
                {
                    "image": str(relative_path),
                    "targets": len(targets),
                    "predictions": len(predictions),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )
            for class_id, x1, y1, x2, y2, confidence in predictions:
                detection_rows.append(
                    {
                        "image": str(relative_path),
                        "class_id": int(class_id),
                        "confidence": f"{confidence:.6f}",
                        "x1": f"{x1:.2f}",
                        "y1": f"{y1:.2f}",
                        "x2": f"{x2:.2f}",
                        "y2": f"{y2:.2f}",
                    }
                )
            completed += 1
            print(
                f"[{completed}/{len(image_paths)}] {relative_path} | "
                f"预测={len(predictions)} TP/FP/FN={tp}/{fp}/{fn}"
            )

    with (OUTPUT_DIR / "逐图统计.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file, fieldnames=["image", "targets", "predictions", "tp", "fp", "fn"]
        )
        writer.writeheader()
        writer.writerows(per_image_rows)
    with (OUTPUT_DIR / "检测框.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["image", "class_id", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        writer.writerows(detection_rows)
    for name, paths in (
        ("无预测图像.txt", no_prediction_images),
        ("存在漏检图像.txt", missed_target_images),
        ("存在误检图像.txt", false_positive_images),
    ):
        (OUTPUT_DIR / name).write_text(
            "\n".join(paths) + ("\n" if paths else ""), encoding="utf-8"
        )

    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    summary = {
        "model_config": str(MODEL_CONFIG),
        "checkpoint": str(CHECKPOINT_PATH),
        "input_dir": str(INPUT_DIR),
        "label_dir": str(LABEL_DIR) if LABEL_DIR else None,
        "model_image_size": MODEL_IMAGE_SIZE,
        "confidence": CONFIDENCE,
        "nms_iou": NMS_IOU,
        "matching_iou": MATCH_IOU,
        "images": len(image_paths),
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "no_prediction_images": len(no_prediction_images),
        "missed_target_images": len(missed_target_images),
        "false_positive_images": len(false_positive_images),
    }
    (OUTPUT_DIR / "汇总.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n========== 小 TIFF 预测与误差分析完成 ==========")
    print(f"TP/FP/FN：{total_tp}/{total_fp}/{total_fn}")
    print(f"Precision/Recall/F1：{precision:.6f}/{recall:.6f}/{f1:.6f}")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
