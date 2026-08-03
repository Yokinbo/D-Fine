"""批量生成负样本的空 YOLO 标签文件。

使用方法：只需要修改下面的两个绝对路径，然后运行本脚本。
负样本图片和标签文件必须同名，例如：
    灵武负样本_001.tif  ->  灵武负样本_001.txt

空 TXT 表示该图片中没有任何目标，转换为 COCO JSON 后会保留图片信息，
但不会生成任何 annotation。
"""

from pathlib import Path


# ============================ 用户配置区 ============================
# 负样本图片所在文件夹（填写图片文件夹的绝对路径）
IMAGE_DIR = Path(r"F:\3能源金三角基础设施识别\数据集优化负样本\1神木与准格尔负样本切片\图片")

# 空 TXT 标签输出文件夹（填写 labels 文件夹的绝对路径）
LABEL_DIR = Path(r"F:\3能源金三角基础设施识别\数据集优化负样本\1神木与准格尔负样本切片\标签")

# 是否递归扫描 IMAGE_DIR 下的子文件夹。普通数据集保持 False。
RECURSIVE = False

# 支持的图片扩展名
IMAGE_SUFFIXES = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
# ====================================================================


def main() -> None:
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"图片文件夹不存在，请修改 IMAGE_DIR：\n{IMAGE_DIR}")

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    pattern = "**/*" if RECURSIVE else "*"
    image_paths = sorted(
        path
        for path in IMAGE_DIR.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    if not image_paths:
        print(f"未找到支持的图片：{IMAGE_DIR}")
        return

    created = 0
    existing_empty = 0
    existing_nonempty = 0

    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        if label_path.exists():
            if label_path.stat().st_size == 0:
                existing_empty += 1
            else:
                # 保护已有标注，避免误覆盖真实目标标签。
                existing_nonempty += 1
            continue

        # 写入零字节文件，表示该负样本没有任何目标框。
        label_path.touch()
        created += 1

    print("\n========== 负样本空标签生成完成 ==========")
    print(f"图片数量：{len(image_paths)}")
    print(f"新生成空 TXT：{created}")
    print(f"原有空 TXT：{existing_empty}")
    print(f"跳过非空 TXT：{existing_nonempty}")
    print(f"标签目录：{LABEL_DIR}")
    print("下一步：将图片复制到 train/images，将这些 TXT 复制到 train/labels，")
    print("然后重新运行 yolo2coco.py 生成 train.json。")


if __name__ == "__main__":
    main()
