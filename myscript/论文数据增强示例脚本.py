"""Create publication-ready examples of D-FINE online augmentation.

This script is intentionally standalone: it reads RGB images and matching YOLO
txt labels, applies augmentation examples used by the training idea, transforms
bounding boxes together with every geometric operation, and writes clean images
plus synchronized YOLO labels for paper figures.

The core small innovation shown here is VRAC: visible-ratio-aware crop. It keeps
the visible area of each large thermal-power-plant object above a threshold, so
online crop augmentation does not create many badly truncated training targets.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]

# ========================= 用户参数配置区 =========================
# 你通常只需要改这里。路径可以写绝对路径，也可以写相对本仓库的路径。
#
# 方式1：只展示一张指定图片。
#   EXAMPLE_IMAGE = "mydatasets/train/images/山西10.tif"
#   EXAMPLE_LABEL = "mydatasets/train/labels/山西10.txt"
#
# 方式2：批量展示一个文件夹。把 EXAMPLE_IMAGE 和 EXAMPLE_LABEL 留空即可。
#   EXAMPLE_IMAGES_DIR = "mydatasets/train/images"
#   EXAMPLE_LABELS_DIR = "mydatasets/train/labels"
#
# 注意：标签必须是 YOLO txt 格式，且文件名要和图片同名。
EXAMPLE_IMAGE = r"E:\YOLO\D-FINE\myscript\增强示例展示\原图\图像\河南48_2.tif"
EXAMPLE_LABEL = r"E:\YOLO\D-FINE\myscript\增强示例展示\原图\标签\河南48_2.txt"
EXAMPLE_IMAGES_DIR = "mydatasets/train/images"
EXAMPLE_LABELS_DIR = "mydatasets/train/labels"
EXAMPLE_OUTPUT_DIR = ""
EXAMPLE_MAX_IMAGES = 3
EXAMPLE_RANDOM_SEED = 20261
DEFAULT_OUTPUT_DIR = r"E:\YOLO\D-FINE\myscript\增强示例展示\增强后r3"
# 论文展示建议保持 False：输出干净影像，不画检测框、不加标题白边。
# 如需检查标签是否同步变化，可改为 True。
DRAW_BOXES_ON_OUTPUT = False

# 是否展示训练配置中的 RandomZoomOut。真实训练配置里包含该算子。
SHOW_RANDOM_ZOOM_OUT = True
SHOW_FORMAT_STEPS = False

# 真实训练逻辑：贴边目标开启边界保护；VRAC 采用随机合法裁剪，不特意选择最明显的一次。
DEMO_VRAC_PROTECT_EDGE_BOXES = True
DEMO_VRAC_SELECT_STRONGEST_CROP = False
# ================================================================


DEFAULT_IMAGES_DIR = REPO_ROOT / "mydatasets" / "train" / "images"
DEFAULT_LABELS_DIR = REPO_ROOT / "mydatasets" / "train" / "labels"

IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}

# These values mirror the proposed crop augmentation in the training config.
VRAC_MIN_VISIBLE_RATIO = 0.7
VRAC_MIN_SCALE = 0.7
VRAC_MAX_SCALE = 1.0
VRAC_TRIALS = 40
EDGE_MARGIN = 2.0
ZOOM_SIDE_RANGE = (1.0, 1.5)
ZOOM_FILL = (114, 114, 114)


def repo_path(path_text: str | Path | None, fallback: Path | None = None) -> Path | None:
    if path_text is None or str(path_text).strip() == "":
        return fallback

    path = Path(path_text)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


@dataclass(frozen=True)
class Box:
    class_id: int
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize synchronized online augmentation for D-FINE and YOLO labels."
    )
    parser.add_argument("--image", type=Path, default=repo_path(EXAMPLE_IMAGE), help="One input image.")
    parser.add_argument("--label", type=Path, default=repo_path(EXAMPLE_LABEL), help="YOLO txt label for --image.")
    parser.add_argument("--images-dir", type=Path, default=repo_path(EXAMPLE_IMAGES_DIR, DEFAULT_IMAGES_DIR))
    parser.add_argument("--labels-dir", type=Path, default=repo_path(EXAMPLE_LABELS_DIR, DEFAULT_LABELS_DIR))
    parser.add_argument("--output-dir", type=Path, default=repo_path(EXAMPLE_OUTPUT_DIR, DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-images", type=int, default=EXAMPLE_MAX_IMAGES)
    parser.add_argument("--seed", type=int, default=EXAMPLE_RANDOM_SEED)
    return parser.parse_args()


def find_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("C:/Windows/Fonts/arial.ttf"),
        Path("C:/Windows/Fonts/calibri.ttf"),
    ]
    for path in candidates:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def filename_text(value: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    output = "".join("_" if char in invalid_chars else char for char in value)
    return output.replace(" ", "")


def load_yolo_boxes(label_path: Path, width: int, height: int) -> List[Box]:
    if not label_path.exists():
        raise FileNotFoundError(f"Matching YOLO label was not found: {label_path}")

    boxes: List[Box] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        values = line.strip().split()
        if not values:
            continue
        if len(values) < 5:
            raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}: {line}")

        class_id = int(values[0])
        center_x, center_y, box_w, box_h = map(float, values[1:5])
        x1 = max(0.0, (center_x - box_w / 2.0) * width)
        y1 = max(0.0, (center_y - box_h / 2.0) * height)
        x2 = min(float(width), (center_x + box_w / 2.0) * width)
        y2 = min(float(height), (center_y + box_h / 2.0) * height)
        if x2 > x1 and y2 > y1:
            boxes.append(Box(class_id, x1, y1, x2, y2))

    if not boxes:
        raise ValueError(f"No valid boxes were read from: {label_path}")
    return boxes


def remote_sensing_color_jitter(
    image: Image.Image, boxes: Sequence[Box], rng: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    brightness = rng.choice([0.85, 1.15])
    contrast = rng.choice([0.80, 1.20])
    saturation = rng.choice([0.85, 1.15])
    hue_shift = rng.choice([-0.02, 0.02])

    output = ImageEnhance.Brightness(image).enhance(brightness)
    output = ImageEnhance.Contrast(output).enhance(contrast)
    output = ImageEnhance.Color(output).enhance(saturation)

    hsv = output.convert("HSV")
    h, s, v = hsv.split()
    shift = round(hue_shift * 255)
    h = h.point(lambda value: (value + shift) % 256)
    output = Image.merge("HSV", (h, s, v)).convert("RGB")

    detail = f"亮度={brightness:.2f}, 对比度={contrast:.2f}, 饱和度={saturation:.2f}"
    return output, list(boxes), detail


def remote_sensing_gaussian_blur(
    image: Image.Image, boxes: Sequence[Box], rng: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    sigma = rng.uniform(0.1, 1.2)
    output = image.filter(ImageFilter.GaussianBlur(radius=sigma))
    return output, list(boxes), f"核大小=3, sigma={sigma:.2f}, 训练概率 p=0.2"


def remote_sensing_gaussian_noise(
    image: Image.Image, boxes: Sequence[Box], rng: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    width, height = image.size
    noise = Image.effect_noise((width, height), 18).convert("L")
    noise_rgb = Image.merge("RGB", (noise, noise, noise))
    output = ImageChops.add(image, noise_rgb, scale=1.8, offset=-58)
    return output, list(boxes), "std≈0.02, 训练概率 p=0.2"


def random_zoom_out(
    image: Image.Image, boxes: Sequence[Box], rng: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    width, height = image.size
    factor = rng.uniform(*ZOOM_SIDE_RANGE)
    output_w = max(width, round(width * factor))
    output_h = max(height, round(height * factor))
    left = rng.randint(0, output_w - width)
    top = rng.randint(0, output_h - height)

    output = Image.new("RGB", (output_w, output_h), ZOOM_FILL)
    output.paste(image, (left, top))
    output_boxes = [
        Box(box.class_id, box.x1 + left, box.y1 + top, box.x2 + left, box.y2 + top)
        for box in boxes
    ]
    return output, output_boxes, f"外扩系数={factor:.2f}, 填充值=114"


def touches_edge(box: Box, width: int, height: int) -> bool:
    return (
        box.x1 <= EDGE_MARGIN
        or box.y1 <= EDGE_MARGIN
        or box.x2 >= width - EDGE_MARGIN
        or box.y2 >= height - EDGE_MARGIN
    )


def visible_box_after_crop(
    box: Box, left: int, top: int, crop_w: int, crop_h: int
) -> Tuple[Box, float]:
    x1 = max(box.x1, left)
    y1 = max(box.y1, top)
    x2 = min(box.x2, left + crop_w)
    y2 = min(box.y2, top + crop_h)
    clipped_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ratio = clipped_area / box.area if box.area > 0 else 0.0
    clipped = Box(box.class_id, x1 - left, y1 - top, x2 - left, y2 - top)
    return clipped, ratio


def vrac(
    image: Image.Image,
    boxes: Sequence[Box],
    rng: random.Random,
    protect_edge_boxes: bool = DEMO_VRAC_PROTECT_EDGE_BOXES,
    select_strongest_crop: bool = DEMO_VRAC_SELECT_STRONGEST_CROP,
) -> Tuple[Image.Image, List[Box], str]:
    width, height = image.size
    if protect_edge_boxes and any(touches_edge(box, width, height) for box in boxes):
        return image.copy(), list(boxes), "边界保护：原目标已接触切片边界"

    best_result: Tuple[int, Image.Image, List[Box], str] | None = None
    for _ in range(VRAC_TRIALS):
        scale = rng.uniform(VRAC_MIN_SCALE, VRAC_MAX_SCALE)
        crop_w = min(width, max(1, round(width * scale)))
        crop_h = min(height, max(1, round(height * scale)))
        left = rng.randint(0, width - crop_w) if crop_w < width else 0
        top = rng.randint(0, height - crop_h) if crop_h < height else 0

        candidates = [visible_box_after_crop(box, left, top, crop_w, crop_h) for box in boxes]
        ratios = [ratio for _, ratio in candidates]
        if ratios and min(ratios) >= VRAC_MIN_VISIBLE_RATIO:
            output = image.crop((left, top, left + crop_w, top + crop_h))
            output_boxes = [box for box, _ in candidates]
            detail = f"最小可见比例={min(ratios):.2f}, 裁剪尺寸={crop_w}x{crop_h}"
            if not select_strongest_crop:
                return output, output_boxes, detail

            crop_area = crop_w * crop_h
            if best_result is None or crop_area < best_result[0]:
                best_result = (crop_area, output, output_boxes, detail)

    if best_result is not None:
        _area, output, output_boxes, detail = best_result
        return output, output_boxes, detail

    return image.copy(), list(boxes), "跳过：40次采样未找到合格裁剪"


def horizontal_flip(
    image: Image.Image, boxes: Sequence[Box], _: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    width, _height = image.size
    output_boxes = [
        Box(box.class_id, width - box.x2, box.y1, width - box.x1, box.y2) for box in boxes
    ]
    return image.transpose(Image.Transpose.FLIP_LEFT_RIGHT), output_boxes, "训练概率 p=0.5"


def vertical_flip(
    image: Image.Image, boxes: Sequence[Box], _: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    _width, height = image.size
    output_boxes = [
        Box(box.class_id, box.x1, height - box.y2, box.x2, height - box.y1) for box in boxes
    ]
    return image.transpose(Image.Transpose.FLIP_TOP_BOTTOM), output_boxes, "训练概率 p=0.5"


def resize_to_640(
    image: Image.Image, boxes: Sequence[Box], _: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    width, height = image.size
    scale_x = 640 / width
    scale_y = 640 / height
    output = image.resize((640, 640), Image.Resampling.BILINEAR)
    output_boxes = [
        Box(
            box.class_id,
            box.x1 * scale_x,
            box.y1 * scale_y,
            box.x2 * scale_x,
            box.y2 * scale_y,
        )
        for box in boxes
    ]
    return output, output_boxes, "必定执行：Resize 到 640x640"


def sanitize_boxes(
    image: Image.Image, boxes: Sequence[Box], _: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    width, height = image.size
    output_boxes = []
    for box in boxes:
        clipped = Box(
            box.class_id,
            max(0.0, min(float(width), box.x1)),
            max(0.0, min(float(height), box.y1)),
            max(0.0, min(float(width), box.x2)),
            max(0.0, min(float(height), box.y2)),
        )
        if clipped.x2 - clipped.x1 >= 1 and clipped.y2 - clipped.y1 >= 1:
            output_boxes.append(clipped)
    removed = len(boxes) - len(output_boxes)
    return image.copy(), output_boxes, f"必定执行：边界框清洗，移除无效框 {removed} 个"


def convert_pil_image_step(
    image: Image.Image, boxes: Sequence[Box], _: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    return image.copy(), list(boxes), "必定执行：转为 float32 并归一化；保存 PNG 时视觉不变"


def convert_boxes_step(
    image: Image.Image, boxes: Sequence[Box], _: random.Random
) -> Tuple[Image.Image, List[Box], str]:
    return image.copy(), list(boxes), "必定执行：边界框转为归一化 cxcywh；图像视觉不变"


def draw_boxes(image: Image.Image, boxes: Sequence[Box]) -> Image.Image:
    output = image.copy()
    draw = ImageDraw.Draw(output)
    line_width = max(3, round(min(output.size) / 160))
    font = find_font(max(16, round(min(output.size) / 28)))

    for index, box in enumerate(boxes, 1):
        coords = (round(box.x1), round(box.y1), round(box.x2), round(box.y2))
        draw.rectangle(coords, outline=(255, 32, 32), width=line_width)
        text = f"HDC {index}"
        text_box = draw.textbbox((0, 0), text, font=font)
        text_w = text_box[2] - text_box[0]
        text_h = text_box[3] - text_box[1]
        label_y = max(0, round(box.y1) - text_h - 8)
        draw.rectangle(
            (round(box.x1), label_y, round(box.x1) + text_w + 8, label_y + text_h + 8),
            fill=(255, 32, 32),
        )
        draw.text((round(box.x1) + 4, label_y + 3), text, fill="white", font=font)
    return output


def make_panel(
    image: Image.Image,
    boxes: Sequence[Box],
    title: str,
    detail: str,
    panel_size: int = 640,
) -> Image.Image:
    annotated = draw_boxes(image, boxes)
    annotated.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)

    title_height = 82
    panel = Image.new("RGB", (panel_size, panel_size + title_height), "white")
    x = (panel_size - annotated.width) // 2
    y = title_height + (panel_size - annotated.height) // 2
    panel.paste(annotated, (x, y))

    draw = ImageDraw.Draw(panel)
    draw.text((16, 9), title, fill=(20, 20, 20), font=find_font(25))
    draw.text((16, 45), detail[:92], fill=(80, 80, 80), font=find_font(17))
    draw.line((0, title_height - 1, panel_size, title_height - 1), fill=(205, 205, 205), width=1)
    return panel


def save_yolo_boxes(label_path: Path, boxes: Sequence[Box], width: int, height: int) -> None:
    lines = []
    for box in boxes:
        center_x = ((box.x1 + box.x2) / 2.0) / width
        center_y = ((box.y1 + box.y2) / 2.0) / height
        box_w = (box.x2 - box.x1) / width
        box_h = (box.y2 - box.y1) / height
        lines.append(
            f"{box.class_id} {center_x:.6f} {center_y:.6f} {box_w:.6f} {box_h:.6f}"
        )
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clear_previous_outputs(image_dir: Path, label_dir: Path) -> None:
    for folder, suffix in [(image_dir, ".png"), (label_dir, ".txt")]:
        if not folder.exists():
            continue
        for path in folder.iterdir():
            if path.is_file() and path.suffix.lower() == suffix:
                path.unlink()


def save_augmented_examples(
    image_path: Path,
    label_path: Path,
    output_dir: Path,
    seed: int,
) -> Path:
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    boxes = load_yolo_boxes(label_path, *image.size)

    operations: List[
        Tuple[
            str,
            str,
            Callable[[Image.Image, Sequence[Box], random.Random], Tuple[Image.Image, List[Box], str]],
        ]
    ] = [
        ("遥感辐射颜色扰动", "RemoteSensingColorJitter：遥感辐射颜色扰动", remote_sensing_color_jitter),
        ("高斯模糊", "RemoteSensingGaussianBlur：高斯模糊", remote_sensing_gaussian_blur),
        ("高斯噪声扰动", "RemoteSensingGaussianNoise：高斯噪声扰动", remote_sensing_gaussian_noise),
        ("可见比例约束裁剪", "VisibleRatioAwareCrop：VRAC可见比例约束裁剪", vrac),
        ("随机水平翻转", "RandomHorizontalFlip：随机水平翻转", horizontal_flip),
        ("随机垂直翻转", "RandomVerticalFlip：随机垂直翻转", vertical_flip),
    ]
    if SHOW_RANDOM_ZOOM_OUT:
        operations.insert(
            3,
            ("随机外扩", "RandomZoomOut：随机外扩", random_zoom_out),
        )
    if SHOW_FORMAT_STEPS:
        operations.extend(
            [
                ("边界框清洗1", "SanitizeBoundingBoxes：边界框清洗", sanitize_boxes),
                ("缩放到640", "Resize：缩放到640x640", resize_to_640),
                ("边界框清洗2", "SanitizeBoundingBoxes：边界框清洗", sanitize_boxes),
                ("图像张量转换", "ConvertPILImage：图像张量转换", convert_pil_image_step),
                ("边界框格式转换", "ConvertBoxes：边界框格式转换", convert_boxes_step),
            ]
        )

    results: List[Tuple[str, Image.Image, List[Box], str]] = [
        ("Original：原始图像", image.copy(), list(boxes), f"{image.width}x{image.height}, 目标数={len(boxes)}")
    ]
    file_names = ["原始图像"]
    for operation_index, (file_name, title, operation) in enumerate(operations, 1):
        operation_rng = random.Random(seed + operation_index * 1009)
        output_image, output_boxes, detail = operation(image.copy(), list(boxes), operation_rng)
        results.append((title, output_image, output_boxes, detail))
        file_names.append(file_name)

    image_dir = output_dir / image_path.stem / "images"
    label_dir = output_dir / image_path.stem / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    clear_previous_outputs(image_dir, label_dir)
    report_lines = [f"图片={image_path}", f"标签={label_path}", f"随机种子={seed}"]
    for index, ((title, output_image, output_boxes, detail), file_name) in enumerate(zip(results, file_names)):
        output_for_save = draw_boxes(output_image, output_boxes) if DRAW_BOXES_ON_OUTPUT else output_image
        output_name = f"{index:02d}_{filename_text(file_name)}"
        output_for_save.save(image_dir / f"{output_name}.png", dpi=(300, 300))
        save_yolo_boxes(label_dir / f"{output_name}.txt", output_boxes, *output_image.size)
        report_lines.append(f"{index:02d} {title}: {detail}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / image_path.stem / "在线数据增强参数说明.txt").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    return image_dir


def collect_inputs(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    if args.image is not None:
        label = args.label or args.labels_dir / f"{args.image.stem}.txt"
        return [(args.image, label)]

    if not args.images_dir.exists():
        raise FileNotFoundError(f"Images directory does not exist: {args.images_dir}")
    image_paths = sorted(
        path for path in args.images_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise FileNotFoundError(f"No supported images were found in: {args.images_dir}")
    return [(path, args.labels_dir / f"{path.stem}.txt") for path in image_paths]


def main() -> None:
    args = parse_args()
    inputs = collect_inputs(args)
    print("Geometric operations transform images and bounding boxes together.")
    print("Photometric jitter changes pixels only; its boxes remain unchanged.")
    for index, (image_path, label_path) in enumerate(inputs):
        output_path = save_augmented_examples(
            image_path,
            label_path,
            args.output_dir,
            seed=args.seed + index * 100_003,
        )
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
