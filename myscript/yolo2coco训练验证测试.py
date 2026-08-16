"""将已划分的 YOLO train/val/test 数据集转换为 COCO JSON。

本脚本只转换标注格式，不重新划分数据集，也不会移动、复制、重命名或删除
任何影像和 YOLO TXT。空 TXT 会作为负样本保留在 COCO ``images`` 中，但
不会产生 ``annotations``。

预期目录结构::

    datasets/mydatasets/
    ├─ train/
    │  ├─ images/
    │  ├─ labels/
    │  └─ classes.txt
    ├─ val/
    │  ├─ images/
    │  ├─ labels/
    │  └─ classes.txt
    └─ test/
       ├─ images/
       ├─ labels/
       └─ classes.txt

运行方式（仓库根目录执行）::

    python myscript/yolo2coco训练验证测试.py

生成文件::

    train/annotations/train.json
    val/annotations/val.json
    test/annotations/test.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]

# =============================================================================
# 用户配置区
# =============================================================================

DATASET_ROOT = REPO_ROOT / "datasets" / "mydatasets"
SPLITS = ("train", "val", "test")

# 支持递归读取 images 子目录；labels 中应存在相同相对路径的同名 TXT。
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# True：每张影像必须有同名 TXT，负样本必须使用空 TXT 明确表示。
# 这样可以避免因为漏放标签文件而把正样本误当成负样本。
REQUIRE_LABEL_FILE = True

# D-FINE 当前配置使用 remap_mscoco_category=False，类别 ID 保持 YOLO 的
# 0-based 编号，因此 COCO categories 也从 0 开始。
CATEGORY_SUPERCATEGORY = "thermal_power_plant"

# =============================================================================


def read_classes(split_root: Path) -> list[str]:
    """读取一个划分的 classes.txt，并拒绝空类别或重复类别。"""
    class_path = split_root / "classes.txt"
    if not class_path.is_file():
        raise FileNotFoundError(f"未找到类别文件：{class_path}")

    classes = [
        line.strip()
        for line in class_path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not classes:
        raise ValueError(f"类别文件为空：{class_path}")
    duplicates = [name for name, count in Counter(classes).items() if count > 1]
    if duplicates:
        raise ValueError(f"类别文件存在重复名称：{class_path} -> {duplicates}")
    return classes


def read_image_size(image_path: Path) -> tuple[int, int]:
    """读取影像宽高，并检查尺寸有效性。"""
    try:
        with Image.open(image_path) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"无法读取影像尺寸：{image_path}") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"影像尺寸无效：{image_path} -> {width}x{height}")
    return int(width), int(height)


def parse_yolo_box(
    line: str,
    *,
    image_width: int,
    image_height: int,
    class_count: int,
    label_path: Path,
    line_number: int,
) -> tuple[int, list[float]]:
    """把一行 ``class cx cy w h`` 转为 COCO ``x y w h``。"""
    values = line.split()
    if len(values) != 5:
        raise ValueError(
            f"标签格式错误：{label_path} 第 {line_number} 行应为 5 列，"
            f"实际为 {len(values)} 列"
        )

    try:
        raw_class_id, center_x, center_y, box_width, box_height = map(float, values)
    except ValueError as exc:
        raise ValueError(f"标签包含非数字内容：{label_path} 第 {line_number} 行") from exc

    class_id = int(raw_class_id)
    if raw_class_id != class_id:
        raise ValueError(f"类别 ID 必须是整数：{label_path} 第 {line_number} 行")
    if not 0 <= class_id < class_count:
        raise ValueError(
            f"类别 ID 越界：{label_path} 第 {line_number} 行为 {class_id}，"
            f"有效范围是 0~{class_count - 1}"
        )
    if not (
        0.0 <= center_x <= 1.0
        and 0.0 <= center_y <= 1.0
        and 0.0 < box_width <= 1.0
        and 0.0 < box_height <= 1.0
    ):
        raise ValueError(
            f"YOLO 归一化坐标超出有效范围：{label_path} 第 {line_number} 行"
        )

    x1 = max(0.0, (center_x - box_width / 2.0) * image_width)
    y1 = max(0.0, (center_y - box_height / 2.0) * image_height)
    x2 = min(float(image_width), (center_x + box_width / 2.0) * image_width)
    y2 = min(float(image_height), (center_y + box_height / 2.0) * image_height)
    clipped_width = x2 - x1
    clipped_height = y2 - y1
    if clipped_width <= 0.0 or clipped_height <= 0.0:
        raise ValueError(f"裁剪后目标框无效：{label_path} 第 {line_number} 行")

    return class_id, [x1, y1, clipped_width, clipped_height]


def find_images(image_dir: Path) -> list[Path]:
    """递归查找支持的影像并按相对路径稳定排序。"""
    return sorted(
        (
            path
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        ),
        key=lambda path: path.relative_to(image_dir).as_posix().casefold(),
    )


def validate_orphan_labels(image_dir: Path, label_dir: Path, image_paths: Sequence[Path]) -> None:
    """拒绝没有对应影像的非空 TXT，避免标签被静默遗漏。"""
    expected = {
        image_path.relative_to(image_dir).with_suffix(".txt").as_posix().casefold()
        for image_path in image_paths
    }
    orphans = []
    for label_path in label_dir.rglob("*.txt"):
        relative = label_path.relative_to(label_dir).as_posix().casefold()
        if relative not in expected and label_path.read_text(encoding="utf-8-sig").strip():
            orphans.append(label_path)
    if orphans:
        preview = "\n".join(f"  - {path}" for path in orphans[:10])
        suffix = f"\n  ...另有 {len(orphans) - 10} 个" if len(orphans) > 10 else ""
        raise FileNotFoundError(f"发现没有对应影像的非空标签：\n{preview}{suffix}")


def convert_split(split: str, classes: Sequence[str]) -> dict[str, int]:
    """转换单个划分并返回统计信息。"""
    split_root = DATASET_ROOT / split
    image_dir = split_root / "images"
    label_dir = split_root / "labels"
    output_json = split_root / "annotations" / f"{split}.json"

    if not image_dir.is_dir():
        raise FileNotFoundError(f"未找到影像目录：{image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"未找到标签目录：{label_dir}")

    image_paths = find_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(f"没有找到支持的影像：{image_dir}")
    validate_orphan_labels(image_dir, label_dir, image_paths)

    coco: dict[str, Any] = {
        "info": {
            "description": f"D-FINE custom dataset - {split}",
            "version": "1.0",
        },
        "licenses": [],
        "categories": [
            {
                "id": class_id,
                "name": class_name,
                "supercategory": CATEGORY_SUPERCATEGORY,
            }
            for class_id, class_name in enumerate(classes)
        ],
        "images": [],
        "annotations": [],
    }

    annotation_id = 1
    negative_images = 0
    missing_label_images = 0
    class_box_counts: Counter[int] = Counter()

    for image_id, image_path in enumerate(image_paths, start=1):
        relative_image = image_path.relative_to(image_dir)
        label_path = label_dir / relative_image.with_suffix(".txt")
        width, height = read_image_size(image_path)

        coco["images"].append(
            {
                "id": image_id,
                "file_name": relative_image.as_posix(),
                "width": width,
                "height": height,
            }
        )

        if not label_path.is_file():
            if REQUIRE_LABEL_FILE:
                raise FileNotFoundError(
                    f"影像缺少同相对路径的标签：{image_path} -> {label_path}"
                )
            missing_label_images += 1
            negative_images += 1
            continue

        lines = [
            line.strip()
            for line in label_path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not lines:
            negative_images += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            class_id, bbox = parse_yolo_box(
                line,
                image_width=width,
                image_height=height,
                class_count=len(classes),
                label_path=label_path,
                line_number=line_number,
            )
            x, y, box_width, box_height = bbox
            coco["annotations"].append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "bbox": bbox,
                    "area": box_width * box_height,
                    "iscrowd": 0,
                    "segmentation": [
                        [
                            x,
                            y,
                            x + box_width,
                            y,
                            x + box_width,
                            y + box_height,
                            x,
                            y + box_height,
                        ]
                    ],
                }
            )
            annotation_id += 1
            class_box_counts[class_id] += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    # Windows 下 faster_coco_eval 可能使用系统默认编码打开 JSON。输出纯 ASCII
    # 可确保中文文件名既不会乱码，json.load 后也能正确还原。
    output_json.write_text(
        json.dumps(coco, ensure_ascii=True, indent=2),
        encoding="ascii",
    )

    return {
        "images": len(coco["images"]),
        "positive_images": len(coco["images"]) - negative_images,
        "negative_images": negative_images,
        "missing_label_images": missing_label_images,
        "annotations": len(coco["annotations"]),
        **{
            f"class_{class_id}_boxes": class_box_counts[class_id]
            for class_id in range(len(classes))
        },
    }


def main() -> None:
    """校验三个划分类别一致性并依次生成 COCO JSON。"""
    split_classes = {split: read_classes(DATASET_ROOT / split) for split in SPLITS}
    reference_classes = split_classes[SPLITS[0]]
    inconsistent = {
        split: classes
        for split, classes in split_classes.items()
        if classes != reference_classes
    }
    if inconsistent:
        details = "\n".join(
            f"  {split}: {classes}" for split, classes in split_classes.items()
        )
        raise ValueError(f"train/val/test 的 classes.txt 不一致：\n{details}")

    print("========== YOLO -> COCO（train / val / test）==========")
    print(f"数据集根目录：{DATASET_ROOT}")
    print(f"类别：{reference_classes}\n")

    all_stats: dict[str, dict[str, int]] = {}
    for split in SPLITS:
        stats = convert_split(split, reference_classes)
        all_stats[split] = stats
        print(
            f"[{split}] 图片 {stats['images']} | 正样本图 {stats['positive_images']} | "
            f"负样本图 {stats['negative_images']} | 目标框 {stats['annotations']}"
        )
        if stats["missing_label_images"]:
            print(f"[{split}] 无标签文件并按负样本处理：{stats['missing_label_images']}")
        for class_id, class_name in enumerate(reference_classes):
            print(
                f"[{split}] class {class_id} ({class_name})："
                f"{stats[f'class_{class_id}_boxes']} 个框"
            )
        print(f"[{split}] JSON：{DATASET_ROOT / split / 'annotations' / f'{split}.json'}\n")

    total_images = sum(stats["images"] for stats in all_stats.values())
    total_boxes = sum(stats["annotations"] for stats in all_stats.values())
    print(f"转换完成：共 {total_images} 张图片、{total_boxes} 个目标框。")
    print("原始影像、YOLO TXT 和既有数据集划分均未修改。")


if __name__ == "__main__":
    main()
