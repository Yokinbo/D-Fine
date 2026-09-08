"""只读验证集分析：python myscript/验证集改进分析.py

修改下面路径。独立选择模型配置，不修改settings.py，不训练、不测速、不执行NMS。
原图/COCO标注/权重只读；每次新建输出子目录。当前明确只支持单类别。
"""
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.ops import box_iou
from src.core import YAMLConfig, yaml_utils
from src.solver.validator import Validator, scale_boxes

# ================= 用户配置：无需修改训练开关 =================
MODEL_CONFIG = ROOT / "configs/dfine/custom/dfine_hgnetv2_m_custom.yml"
CHECKPOINT_PATH = Path(r"G:\b1\权重结果\改进实验dfine\base\base_2e-4_SD3407\best_map50.pth")
VAL_IMAGES_DIR = ROOT / "datasets/mydatasets/val/images"
VAL_ANNOTATION = ROOT / "datasets/mydatasets/val/annotations/val.json"
OUTPUT_DIR = Path(r"G:\b1\权重结果\改进实验dfine\base\base_2e-4_SD3407\SD3407验证集分析")
INPUT_SIZE = 512
BATCH_SIZE = 6
NUM_WORKERS = 0
DEVICE = "cuda"
CONFIDENCE = 0.5
MATCH_IOU = 0.5
SAVE_ALL_PREVIEWS = False  # 默认只保存有FP/FN的图；绿色TP、红色FP、橙色FN
# ============================================================


def inside(path, parent):
    return Path(path).resolve().is_relative_to(Path(parent).resolve())


def check_output(output, protected):
    for path in protected:
        if inside(output, path) or inside(path, output):
            raise ValueError(f"输出目录与保护路径重叠：{output} / {path}")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def match_details(pred, gt, confidence=0.5, iou=0.5):
    """与Validator单类别固定阈值指标一致：IoU降序贪心匹配。"""
    indices = torch.where(pred["scores"] >= confidence)[0]
    boxes = pred["boxes"][indices]
    overlaps = box_iou(boxes, gt["boxes"])
    p, g = torch.where(overlaps >= iou)
    used_p, used_g = set(), set()
    for k in torch.argsort(-overlaps[p, g]).tolist():
        pi, gi = int(p[k]), int(g[k])
        if pi not in used_p and gi not in used_g:
            used_p.add(pi)
            used_g.add(gi)
    return indices, used_p, used_g, overlaps


def error_rows(image_id, pred, gt):
    indices, mp, mg, overlaps = match_details(pred, gt, CONFIDENCE, MATCH_IOU)
    rows = []
    for local, original in enumerate(indices.tolist()):
        if local in mp:
            kind = "TP"
        else:
            best = float(overlaps[local].max()) if overlaps.shape[1] else 0.0
            kind = "duplicate_candidate" if best >= MATCH_IOU else (
                "localization_candidate" if best >= 0.1 else "background_candidate")
        rows.append(dict(image_id=image_id, type=kind, index=original,
                         score=float(pred["scores"][original]),
                         box=pred["boxes"][original].tolist()))
    low = pred["boxes"][pred["scores"] < CONFIDENCE]
    for gi in range(len(gt["boxes"])):
        if gi not in mg:
            low_hit = len(low) and bool((box_iou(low, gt["boxes"][gi:gi+1]) >= MATCH_IOU).any())
            rows.append(dict(image_id=image_id, type="FN_low_score_candidate" if low_hit else "FN_other",
                             index=gi, score=None, box=gt["boxes"][gi].tolist()))
    return rows, len(mp), len(indices)-len(mp), len(gt["boxes"])-len(mg)


def main():
    for path in (MODEL_CONFIG, CHECKPOINT_PATH, VAL_ANNOTATION):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    if not VAL_IMAGES_DIR.is_dir():
        raise FileNotFoundError(VAL_IMAGES_DIR)
    check_output(OUTPUT_DIR, [ROOT / "datasets", VAL_IMAGES_DIR.parent,
                             VAL_ANNOTATION.parent, CHECKPOINT_PATH, MODEL_CONFIG])
    if not 0 <= CONFIDENCE <= 1 or not 0 < MATCH_IOU <= 1:
        raise ValueError("阈值不合法")
    annotation = json.loads(VAL_ANNOTATION.read_text(encoding="utf-8"))
    if len(annotation["categories"]) != 1:
        raise ValueError("此分析脚本只支持单类别数据")
    # 缺图/越界路径提前失败，不能悄悄把样本忽略掉。
    images = {x["id"]: x for x in annotation["images"]}
    for info in images.values():
        path = VAL_IMAGES_DIR / info["file_name"]
        if not inside(path, VAL_IMAGES_DIR) or not path.is_file():
            raise ValueError(f"图片路径无效：{path}")
    cfg_dict = yaml_utils.load_config(str(MODEL_CONFIG))
    cfg_dict.update(num_classes=1, remap_mscoco_category=False,
                    eval_spatial_size=[INPUT_SIZE, INPUT_SIZE])
    cfg_dict["HGNetv2"]["pretrained"] = False
    loader = cfg_dict["val_dataloader"]
    loader.update(total_batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=False, drop_last=False)
    loader["dataset"].update(img_folder=str(VAL_IMAGES_DIR), ann_file=str(VAL_ANNOTATION))
    for op in loader["dataset"]["transforms"].get("ops", []):
        if op.get("type") == "Resize":
            op["size"] = [INPUT_SIZE, INPUT_SIZE]
    cfg = YAMLConfig(str(MODEL_CONFIG), **cfg_dict)
    device = torch.device(DEVICE)
    model = cfg.model.to(device).eval()
    checkpoint = torch.load(str(CHECKPOINT_PATH), map_location="cpu")
    if "ema" in checkpoint and checkpoint["ema"] is not None:
        state, source = checkpoint["ema"]["module"], "ema.module"
    elif "model" in checkpoint:
        state, source = checkpoint["model"], "model"
    else:
        state, source = checkpoint, "state_dict"
    model.load_state_dict(state, strict=True)
    postprocessor = cfg.postprocessor.to(device).eval()
    data_loader = cfg.val_dataloader
    evaluator = cfg.evaluator
    evaluator.cleanup()
    run = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run.mkdir(parents=True, exist_ok=False)
    (run / "previews").mkdir()
    write_json(run / "resolved_config.json", cfg_dict)
    print(f"只读分析：{MODEL_CONFIG}\n权重：{CHECKPOINT_PATH} ({source})\n输出：{run}")
    all_gt, all_pred, records, errors, per_image = [], [], [], [], []
    seen = set()
    with torch.inference_mode():
        for samples, targets in data_loader:
            samples = samples.to(device)
            sizes = torch.stack([t["orig_size"] for t in targets]).to(device)
            results = postprocessor(model(samples), sizes)  # 与valid.py一致：FP32、非deploy、无NMS
            evaluator.update({t["image_id"].item(): r for t, r in zip(targets, results)})
            for index, (target, result) in enumerate(zip(targets, results)):
                image_id = int(target["image_id"].item())
                if image_id in seen:
                    raise RuntimeError("重复image_id")
                seen.add(image_id)
                gt = {"boxes": scale_boxes(target["boxes"],
                      (target["orig_size"][1], target["orig_size"][0]),
                      (samples[index].shape[-1], samples[index].shape[-2])).cpu(),
                      "labels": target["labels"].cpu()}
                pred = {k: result[k].detach().cpu() for k in ("boxes", "labels", "scores")}
                if any(len(x["labels"]) and bool((x["labels"] != 0).any()) for x in (gt, pred)):
                    raise ValueError("固定阈值分析要求单类别标签ID=0，与本仓库valid.py口径一致")
                all_gt.append(gt)
                all_pred.append(pred)
                rows, tp, fp, fn = error_rows(image_id, pred, gt)
                # 防止未来Validator修改后，明细与正式计数悄悄偏离。
                official = Validator([gt], [pred], CONFIDENCE, MATCH_IOU).compute_metrics()
                assert (tp, fp, fn) == tuple(official[k] for k in ("TPs", "FPs", "FNs"))
                info = images[image_id]
                per_image.append(dict(image_id=image_id, file_name=info["file_name"], tp=tp, fp=fp, fn=fn))
                errors.extend(rows)
                records.append(dict(image_id=image_id, file_name=info["file_name"],
                                    gt={k:v.tolist() for k,v in gt.items()},
                                    predictions={k:v.tolist() for k,v in pred.items()}))
                if SAVE_ALL_PREVIEWS or fp or fn:
                    with Image.open(VAL_IMAGES_DIR / info["file_name"]) as original:
                        preview = original.convert("RGB")
                    draw = ImageDraw.Draw(preview)
                    for row in rows:
                        color = "lime" if row["type"] == "TP" else ("orange" if row["type"].startswith("FN") else "red")
                        draw.rectangle(row["box"], outline=color, width=2)
                        draw.text(tuple(row["box"][:2]), row["type"] + (f" {row['score']:.3f}" if row["score"] is not None else ""), fill=color)
                    preview.save(run / "previews" / f"{image_id}.png")
            print(f"已分析 {len(seen)}/{len(images)} 张")
    if seen != set(images):
        raise RuntimeError("处理图片集合与COCO标注不一致")
    validator = Validator(all_gt, all_pred, CONFIDENCE, MATCH_IOU)
    fixed = validator.compute_metrics()
    curve = validator.compute_confidence_curve()
    write_json(run / "predictions_all.json", records)  # 未按0.5截断；保留官方top-k全部候选
    write_json(run / "error_details.json", errors)
    write_csv(run / "per_image.csv", per_image, ["image_id", "file_name", "tp", "fp", "fn"])
    for name, key in (("false_positive_images.txt", "fp"), ("missed_images.txt", "fn")):
        (run / name).write_text("\n".join(r["file_name"] for r in per_image if r[key]), encoding="utf-8")
    (run / "README.txt").write_text(
        "原图、COCO标注、权重未修改。每次运行新建目录。\n"
        "per_image.csv: 每图TP/FP/FN；error_details.json: 逐框候选错误类别。\n"
        "previews: 绿色=正确预测TP，红色=误检预测FP，橙色=漏检真值FN。文件名为COCO image_id。\n"
        "predictions_all.json: 官方后处理top-k全部分数及真值，不执行NMS或0.5截断。\n"
        "curves.png / coco_pr50.csv: COCO标准PR (IoU=0.5, maxDets=100)。\n"
        "confidence_metrics.csv: 与valid.py相同的IoU优先匹配，阈值0~1步长0.01。\n"
        "不要通过该阈值曲线积分代替COCO AP。最佳阈值仅探索，不替代固定0.5论文指标。\n"
        "background/localization/duplicate/low_score都是几何启发式候选，需人工核验。\n"
        "summary.json记录权重与标注SHA256、指标。resolved_config.json记录合并配置。\n",
        encoding="utf-8")
    write_csv(run / "confidence_metrics.csv", curve["curve"], list(curve["curve"][0]))
    evaluator.synchronize_between_processes()
    evaluator.accumulate()
    evaluator.summarize()
    coco = evaluator.coco_eval["bbox"]
    # COCO标准PR：IoU0.5、area=all、maxDets最后一档；不是固定阈值Validator曲线的积分。
    ti = int(np.where(np.isclose(coco.params.iouThrs, 0.5))[0][0])
    precision = coco.eval["precision"][ti, :, 0, 0, -1]
    valid = precision >= 0
    pr_rows = [dict(recall=float(r), precision=float(p)) for r,p in zip(coco.params.recThrs[valid], precision[valid])]
    write_csv(run / "coco_pr50.csv", pr_rows, ["recall", "precision"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot([r["recall"] for r in pr_rows], [r["precision"] for r in pr_rows])
    axes[0].set(xlabel="Recall", ylabel="Precision", title="COCO PR @ IoU=0.50")
    for metric in ("precision", "recall", "f1"):
        axes[1].plot([r["confidence"] for r in curve["curve"]], [r[metric] for r in curve["curve"]], label=metric)
    axes[1].axvline(CONFIDENCE, color="gray", linestyle="--")
    axes[1].set(xlabel="Confidence", ylabel="Metric", title="Validator threshold sweep")
    axes[1].legend()
    for axis in axes:
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1.01)
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run / "curves.png", dpi=180)
    plt.close(fig)
    write_json(run / "summary.json", dict(config=str(MODEL_CONFIG), checkpoint=str(CHECKPOINT_PATH),
        checkpoint_sha256=digest(CHECKPOINT_PATH), annotation_sha256=digest(VAL_ANNOTATION),
        weight_source=source, fixed_metrics=fixed, confidence_analysis=curve,
        coco_stats=coco.stats.tolist(), images=len(seen),
        note="错误类别为启发式候选，需人工复核。低分仅指官方top-k候选；不等于全部query。阈值分析不替换固定0.5指标。"))
    print(f"完成：{fixed}\n结果：{run}")


if __name__ == "__main__":
    main()
