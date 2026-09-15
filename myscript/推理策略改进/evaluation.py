"""整景单类别评价；真值仅供离线评价，不参与窗口选择。

AP 使用所有传入预测的分数排序和 101 点插值，不应用最终展示置信度。
本模块不实现 COCO 的 maxDets、面积分组、ignore 或 crowd 规则，因此指标
明确命名为 scene_single_class_*，不能直接标成官方 COCO AP。
实现仅依赖 Python 标准库，允许脱离模型、CUDA 和地理数据运行单元测试。
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


Box = Tuple[float, float, float, float]


@dataclass(frozen=True)
class _Record:
    box: Box
    class_id: int
    index: int
    score: float = 1.0
    plant_id: str = ""


def _field(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _box(item: Any) -> Box:
    values = _field(item, "box")
    if values is None or len(values) != 4:
        raise ValueError("每条记录必须提供四个 xyxy 坐标：.box 或 ['box']。")
    result = tuple(float(value) for value in values)
    if not all(math.isfinite(value) for value in result):
        raise ValueError("检测框坐标必须有限。")
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("检测框必须具有正的宽度和高度。")
    return result  # type: ignore[return-value]


def _records(items: Iterable[Any], *, is_prediction: bool) -> List[_Record]:
    result = []
    for index, item in enumerate(items):
        score = float(_field(item, "score", 1.0)) if is_prediction else 1.0
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError("预测分数必须位于 [0, 1]。")
        result.append(_Record(
            _box(item), int(_field(item, "class_id", 1)), index, score,
            str(_field(item, "plant_id", "GT%06d" % index)),
        ))
    return result


def _iou(first: Box, second: Box) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    area_first = (first[2] - first[0]) * (first[3] - first[1])
    area_second = (second[2] - second[0]) * (second[3] - second[1])
    return intersection / (area_first + area_second - intersection)


def _match(
    ranked: Sequence[_Record], ground_truths: Sequence[_Record], threshold: float,
) -> List[Tuple[_Record, Optional[int], float]]:
    """按分数排序后，仅在尚未匹配且类别一致的真值中选最佳框。"""
    unmatched = set(range(len(ground_truths)))
    matches = []
    for prediction in ranked:
        best_index = None
        best_overlap = -1.0
        # 固定索引顺序使 IoU 相同的情形可复现。
        for index in sorted(unmatched):
            if ground_truths[index].class_id != prediction.class_id:
                continue
            overlap = _iou(prediction.box, ground_truths[index].box)
            if overlap > best_overlap:
                best_index, best_overlap = index, overlap
        if best_index is not None and best_overlap >= threshold:
            unmatched.remove(best_index)
            matches.append((prediction, best_index, best_overlap))
        else:
            matches.append((prediction, None, 0.0))
    return matches


def _average_precision(
    ranked: Sequence[_Record], ground_truths: Sequence[_Record], threshold: float,
) -> Optional[float]:
    if not ground_truths:
        return None
    matches = _match(ranked, ground_truths, threshold)
    if not matches:
        return 0.0
    recalls, precisions = [], []
    true_positives = 0
    for index, (_, gt_index, _) in enumerate(matches):
        true_positives += gt_index is not None
        recalls.append(true_positives / len(ground_truths))
        precisions.append(true_positives / (index + 1))
    # 右侧精度包络；固定 101 个召回采样点，包括 0 和 1。
    for index in range(len(precisions) - 2, -1, -1):
        precisions[index] = max(precisions[index], precisions[index + 1])
    recall_index = 0
    total = 0.0
    for level in range(101):
        while recall_index < len(recalls) and recalls[recall_index] < level / 100.0:
            recall_index += 1
        if recall_index < len(precisions):
            total += precisions[recall_index]
    return total / 101.0


def _max_axis_visible(start: float, end: float, extent: int, size: int, offset: int) -> float:
    """真正非重叠的格网，边缘格裁到图像范围；不将末窗回退并制造重叠。"""
    visible_start, visible_end = max(0.0, start), min(float(extent), end)
    if visible_end <= visible_start:
        return 0.0
    first = math.floor((visible_start - offset) / size)
    last = math.ceil((visible_end - offset) / size) - 1
    best = 0.0
    for cell in range(first, last + 1):
        low = max(0.0, offset + cell * size)
        high = min(float(extent), offset + (cell + 1) * size)
        best = max(best, max(0.0, min(visible_end, high) - max(visible_start, low)))
    return best


def _strata(
    ground_truths: Sequence[_Record], matches: Mapping[int, Tuple[_Record, float]],
    width: int, height: int, base_size: int, reference_offset: Tuple[int, int],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    grouped: Dict[str, List[int]] = {name: [] for name in (
        "all", "oversized", "position_only_truncated", "contained",
        "internal_oversized", "internal_position_only_truncated",
        "internal_visible_lt_0_9", "internal_visible_lt_0_7", "image_outer_boundary_touch",
    )}
    rows = []
    for index, gt in enumerate(ground_truths):
        x1, y1, x2, y2 = gt.box
        box_width, box_height = x2 - x1, y2 - y1
        visible = _max_axis_visible(x1, x2, width, base_size, reference_offset[0]) * (
            _max_axis_visible(y1, y2, height, base_size, reference_offset[1])
        ) / (box_width * box_height)
        outer_touch = x1 <= 1.0 or y1 <= 1.0 or x2 >= width - 1.0 or y2 >= height - 1.0
        oversized = box_width > base_size or box_height > base_size
        category = "oversized" if oversized else (
            "position_only_truncated" if visible < 1.0 - 1e-9 else "contained"
        )
        grouped["all"].append(index)
        grouped[category].append(index)
        if outer_touch:
            grouped["image_outer_boundary_touch"].append(index)
        else:
            if category != "contained":
                grouped["internal_" + category].append(index)
            if visible < 0.9:
                grouped["internal_visible_lt_0_9"].append(index)
            if visible < 0.7:
                grouped["internal_visible_lt_0_7"].append(index)
        matched = matches.get(index)
        rows.append({
            "ground_truth_index": index, "plant_id": gt.plant_id, "box": list(gt.box),
            "max_visible_ratio": visible, "truncation_category": category,
            "image_outer_boundary_touch": outer_touch, "matched": matched is not None,
            "prediction_index": matched[0].index if matched else None,
            "matched_iou": matched[1] if matched else None,
        })
    strata: Dict[str, Any] = {
        "base_size": base_size, "reference_offset_xy": list(reference_offset),
        "grid_definition": "fixed_nonoverlapping_grid_clipped_at_image_boundary",
        "outer_boundary_tolerance_px": 1.0,
        "internal_definition": "GT 不贴整景外边界；max_visible_ratio 为单格最大相交面积/GT面积。",
    }
    for name, indices in grouped.items():
        true_positives = sum(index in matches for index in indices)
        strata[name] = {
            "gt_count": len(indices), "tp": true_positives,
            "recall": true_positives / len(indices) if indices else None,
        }
    return strata, rows


def _edge_errors(
    ground_truths: Sequence[_Record], matches: Mapping[int, Tuple[_Record, float]],
) -> Dict[str, Any]:
    errors: List[Tuple[float, ...]] = []
    for gt_index, (prediction, _) in matches.items():
        truth = ground_truths[gt_index].box
        width, height = truth[2] - truth[0], truth[3] - truth[1]
        denominators = (width, height, width, height)
        errors.append(tuple((prediction.box[i] - truth[i]) / denominators[i] for i in range(4)))
    absolute, signed = {}, {}
    for index, name in enumerate(("left", "top", "right", "bottom")):
        absolute[name] = sum(abs(row[index]) for row in errors) / len(errors) if errors else None
        signed[name] = sum(row[index] for row in errors) / len(errors) if errors else None
    return {
        "matched_gt_count": len(errors),
        "normalization": "left/right by GT width; top/bottom by GT height",
        "scope": "仅固定置信度与匹配IoU下的TP；不代表漏检目标的定位误差。",
        "signed_definition": "prediction_coordinate - ground_truth_coordinate",
        "mean_absolute": absolute, "mean_signed": signed,
        "mean_absolute_all_edges": sum(abs(v) for row in errors for v in row) / (4 * len(errors))
        if errors else None,
    }


def evaluate_predictions(
    predictions: Iterable[Any], ground_truths: Iterable[Any], *, width: int, height: int,
    base_size: int = 512, reference_offset: Tuple[int, int] = (0, 0),
    final_confidence: float = 0.5, match_iou: float = 0.5,
    valid_area_km2: Optional[float] = None,
) -> Dict[str, Any]:
    """评价融合后的所有预测，返回可用 ``json.dumps(..., allow_nan=False)`` 写出的字典。

    预测需提供 box/score/class_id，真值提供 box/plant_id；class_id 缺省为 1。
    对象属性和字典均可。仅支持一个真值类别；其他类别预测计为误检。
    AP 不设最大检测数，使用所有传入分数；P/R/F1、重复框和分层召回应用
    final_confidence。无真值时 AP/Recall/F1 为 None；无预测时 Precision 为 0。
    reference_offset 应在不同推理方法及不同推理网格起点实验中保持相同。
    外边界贴边只描述观测几何，不证明实际厂区被截断。
    """
    for name, value in (("width", width), ("height", height), ("base_size", base_size)):
        if not math.isfinite(float(value)) or int(value) != value or value <= 0:
            raise ValueError(name + " 必须为正整数。")
    width, height, base_size = int(width), int(height), int(base_size)
    if len(reference_offset) != 2 or any(
        not math.isfinite(float(v)) or int(v) != v for v in reference_offset
    ):
        raise ValueError("reference_offset 必须包含两个整数。")
    reference_offset = tuple(int(v) % base_size for v in reference_offset)  # type: ignore[assignment]
    if not 0.0 <= final_confidence <= 1.0 or not 0.0 < match_iou <= 1.0:
        raise ValueError("final_confidence 应位于 [0,1]，match_iou 应位于 (0,1]。")
    if valid_area_km2 is not None:
        valid_area_km2 = float(valid_area_km2)
        if not math.isfinite(valid_area_km2) or valid_area_km2 <= 0.0:
            raise ValueError("valid_area_km2 必须为正的有限值，未知时传 None。")
    predicted, truths = _records(predictions, is_prediction=True), _records(ground_truths, is_prediction=False)
    if len({gt.class_id for gt in truths}) > 1:
        raise ValueError("本评价器只支持单类别真值，不能用于多类别 mAP。")
    ranked = sorted(predicted, key=lambda item: (-item.score, item.index))
    selected = [item for item in ranked if item.score >= final_confidence]
    assignments = _match(selected, truths, match_iou)
    matched = {index: (prediction, overlap) for prediction, index, overlap in assignments if index is not None}
    tp, fp, fn = len(matched), len(selected) - len(matched), len(truths) - len(matched)
    precision = tp / len(selected) if selected else 0.0
    recall = tp / len(truths) if truths else None
    f1 = 2 * precision * recall / (precision + recall) if recall is not None and precision + recall else (
        0.0 if recall is not None else None
    )
    # 每个未匹配预测最多计一次。已分配给另一个真值的框不重复计数。
    duplicates = sum(
        any(prediction.class_id == truths[index].class_id and _iou(prediction.box, truths[index].box) >= match_iou
            for index in matched)
        for prediction, gt_index, _ in assignments if gt_index is None
    )
    thresholds = [value / 100.0 for value in range(50, 96, 5)]
    average_precisions = [_average_precision(ranked, truths, threshold) for threshold in thresholds]
    strata, per_gt = _strata(truths, matched, width, height, base_size, reference_offset)
    return {
        "scene_single_class_AP50_101pt": average_precisions[0],
        "scene_single_class_AP75_101pt": average_precisions[5],
        "scene_single_class_mAP50_95_101pt": sum(average_precisions) / len(average_precisions) if truths else None,
        "scene_single_class_AP_by_iou": {"%.2f" % threshold: ap for threshold, ap in zip(thresholds, average_precisions)},
        "precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
        "num_ground_truths": len(truths), "num_predictions_total": len(predicted),
        "num_predictions_selected": len(selected), "final_confidence": final_confidence, "match_iou": match_iou,
        "valid_area_km2": valid_area_km2, "fp_per_km2": fp / valid_area_km2 if valid_area_km2 is not None else None,
        "duplicate_boxes": duplicates, "edge_errors_tp_only": _edge_errors(truths, matched),
        "reference_grid_strata": strata, "per_ground_truth": per_gt,
        "protocol": {
            "ap": "整景单类别，101点插值，无maxDets上限；不是官方COCO评价。",
            "ap_input": "所有传入预测，未应用final_confidence；上游候选阈值仍会限制召回。",
            "matching": "score-descending greedy best-unmatched-GT IoU, class-aware; stable input-order ties",
            "duplicates": "固定阈值下，与已匹配同类GT达到match_iou的未匹配预测；每个预测最多计一次。",
            "ground_truth_use": "offline_evaluation_only",
            "unsupported_coco_features": ["maxDets", "area_ranges", "ignore", "iscrowd"],
        },
    }
