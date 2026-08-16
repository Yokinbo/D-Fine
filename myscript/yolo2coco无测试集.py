"""将已划分好的 YOLO 数据集转换为 D-FINE 所需的 COCO JSON。

本脚本只做格式转换：
* 不随机划分训练/验证集；
* 不移动、复制或删除任何图片和标签；
* 负样本对应的空 TXT 会写入 images，但不会写入 annotations。

目录结构：
    train/images, train/labels, train/classes.txt
    val/images,   val/labels,   val/classes.txt

运行：
    python myscript/yolo2coco.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from PIL import Image


# =============================================================================
# 用户配置区：如后续更换数据集，只修改这里的绝对路径
# =============================================================================
TRAIN_ROOT = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\train")
VAL_ROOT = Path(r"E:\YOLO\D-FINE\datasets\mydatasets\val")

TRAIN_JSON = TRAIN_ROOT / "annotations" / "train.json"
VAL_JSON = VAL_ROOT / "annotations" / "val.json"

# 支持的影像格式；其他文件会被自动忽略。
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
# =============================================================================


def read_classes(dataset_root: Path) -> List[str]:
    class_file = dataset_root / "classes.txt"
    if not class_file.is_file():
        raise FileNotFoundError(f"未找到类别文件：{class_file}")
    classes = [line.strip() for line in class_file.read_text(encoding="utf-8-sig").splitlines()]
    classes = [name for name in classes if name]
    if not classes:
        raise ValueError(f"classes.txt 为空：{class_file}")
    return classes


def yolo_line_to_coco(
    line: str,
    image_width: int,
    image_height: int,
    class_count: int,
    label_path: Path,
    line_number: int,
) -> Tuple[int, List[float]]:
    values = line.split()
    if len(values) != 5:
        raise ValueError(f"标签格式错误：{label_path} 第 {line_number} 行应有 5 列，实际为 {len(values)} 列")

    class_id = int(values[0])
    if not 0 <= class_id < class_count:
        raise ValueError(f"类别编号越界：{label_path} 第 {line_number} 行的类别为 {class_id}")

    center_x, center_y, width, height = map(float, values[1:])
    if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
        raise ValueError(f"YOLO 坐标超出范围：{label_path} 第 {line_number} 行")

    x1 = max(0.0, (center_x - width / 2.0) * image_width)
    y1 = max(0.0, (center_y - height / 2.0) * image_height)
    x2 = min(float(image_width), (center_x + width / 2.0) * image_width)
    y2 = min(float(image_height), (center_y + height / 2.0) * image_height)
    box_width = x2 - x1
    box_height = y2 - y1
    if box_width <= 0 or box_height <= 0:
        raise ValueError(f"无效目标框：{label_path} 第 {line_number} 行")
    return class_id, [x1, y1, box_width, box_height]


def convert_split(dataset_root: Path, output_json: Path, classes: Sequence[str]) -> Dict[str, int]:
    image_dir = dataset_root / "images"
    label_dir = dataset_root / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(f"数据集必须包含 images 和 labels 文件夹：{dataset_root}")

    image_paths = sorted(
        path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"未在图片文件夹找到支持的影像：{image_dir}")

    dataset: Dict[str, list] = {
        "categories": [
            {"id": index, "name": name, "supercategory": "fire_power_plant"}
            for index, name in enumerate(classes)
        ],
        "images": [],
        "annotations": [],
    }
    annotation_id = 1
    negative_images = 0

    for image_id, image_path in enumerate(image_paths, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"
        # 强制每张图都有 TXT；负样本的 TXT 必须是空文件，避免正样本漏标被误认为负样本。
        if not label_path.is_file():
            raise FileNotFoundError(f"图片缺少同名标签 TXT：{image_path.name} -> {label_path}")

        with Image.open(image_path) as image:
            image_width, image_height = image.size
        dataset["images"].append(
            {"id": image_id, "file_name": image_path.name, "width": image_width, "height": image_height}
        )

        lines = [line.strip() for line in label_path.read_text(encoding="utf-8-sig").splitlines()]
        lines = [line for line in lines if line]
        if not lines:
            negative_images += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            class_id, bbox = yolo_line_to_coco(
                line, image_width, image_height, len(classes), label_path, line_number
            )
            x, y, width, height = bbox
            dataset["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": width * height,
                    "iscrowd": 0,
                    "segmentation": [[x, y, x + width, y, x + width, y + height, x, y + height]],
                }
            )
            annotation_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    # faster_coco_eval 在 Windows 上会按系统默认编码（通常为 GBK）读取 JSON。
    # 因此使用 ensure_ascii=True，使中文文件名以 \uXXXX 转义写入，文件本身仅含 ASCII，
    # 可同时被 UTF-8 与 GBK 正确读取；json.load 后仍会还原为原始中文文件名。
    output_json.write_text(json.dumps(dataset, ensure_ascii=True, indent=2), encoding="ascii")
    return {
        "images": len(dataset["images"]),
        "negative_images": negative_images,
        "annotations": len(dataset["annotations"]),
    }


def main() -> None:
    train_classes = read_classes(TRAIN_ROOT)
    val_classes = read_classes(VAL_ROOT)
    if train_classes != val_classes:
        raise ValueError(f"训练/验证 classes.txt 不一致：\ntrain={train_classes}\nval={val_classes}")

    print("========== YOLO -> COCO 格式转换 ==========")
    print(f"训练集：{TRAIN_ROOT}")
    train_stats = convert_split(TRAIN_ROOT, TRAIN_JSON, train_classes)
    print(
        f"训练集完成：图片 {train_stats['images']} | 空标签负样本 {train_stats['negative_images']} | "
        f"目标框 {train_stats['annotations']}\nJSON：{TRAIN_JSON}"
    )

    print(f"\n验证集：{VAL_ROOT}")
    val_stats = convert_split(VAL_ROOT, VAL_JSON, val_classes)
    print(
        f"验证集完成：图片 {val_stats['images']} | 空标签负样本 {val_stats['negative_images']} | "
        f"目标框 {val_stats['annotations']}\nJSON：{VAL_JSON}"
    )
    print("\n转换完成。未移动或修改任何图片、TXT 标签和数据集划分。")


if __name__ == "__main__":
    main()
