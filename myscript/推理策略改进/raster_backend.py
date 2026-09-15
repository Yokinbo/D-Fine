"""大图 RGB 读取、D-FINE 适配与结果导出；不修改现有模型和推理脚本。

此模块在实际运行时才加载重依赖，策略核心可在没有 torch/rasterio 时测试。
"""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

from strategy import Observation

REPO_ROOT = Path(__file__).resolve().parents[2]


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_inference_core(args, config):
    """复用原模型加载器；load_state_dict 默认 strict=True，配置显式指定。"""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    path = REPO_ROOT / "myscript" / "大范围遥感影像火电厂推理策略对比实验.py"
    name = "crop_response_dfine_core"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.MODEL_CONFIG = str(Path(args.config).resolve())
    module.CHECKPOINT = str(Path(args.checkpoint).resolve())
    module.NUM_CLASSES = 1
    module.DEVICE = args.device
    module.USE_AMP = not args.no_amp
    module.MODEL_INPUT_SIZE = config.model_input_size
    module.BASE_TILE_SIZE = config.base_size
    module.CANDIDATE_CONF = config.candidate_confidence
    module.MAX_CANDIDATES_PER_WINDOW_VIEW = args.topk
    module.BATCH_SIZE = args.batch_size
    module.BLACK_THRESHOLD = args.black_threshold
    module.MIN_VALID_RATIO = args.min_valid_ratio
    module.ALLOW_NON_UINT8 = False
    module.PREVIEW_MAX_SIZE = args.preview_max_size
    module.require_runtime_dependencies()
    return module


class RasterBackend:
    """所有观测都使用原图像素坐标；同一 batch 中窗口必须同尺寸。"""

    def __init__(self, src, model, device, core, config, *, batch_size=6,
                 topk=50, black_threshold=3, min_valid_ratio=0.2):
        self.src, self.model, self.device, self.core, self.config = src, model, device, core, config
        self.batch_size, self.topk = batch_size, topk
        self.black_threshold, self.min_valid_ratio = black_threshold, min_valid_ratio
        if src.count < 3 or any(dtype != "uint8" for dtype in src.dtypes[:3]):
            raise ValueError("输入必须至少含三个 uint8 RGB 波段；请先按训练影像的辐射范围转换。")
        if src.crs is None:
            raise ValueError("输入影像缺少 CRS，无法可靠导出地理坐标；请先补充真实空间参考。")
        self.stats = {
            "base_attempted_windows": 0, "refine_attempted_windows": 0,
            "base_skipped_windows": 0, "refine_skipped_windows": 0,
            "base_model_views": 0, "refine_model_views": 0,
            "model_forward_calls": 0, "model_view_count": 0,
            "discarded_invalid_predictions": 0, "clipped_predictions": 0,
            "discarded_by_topk": 0, "discarded_on_nodata_center": 0,
            "read_seconds": 0.0, "model_seconds": 0.0,
        }
        self.window_diagnostics = []

    def synchronize(self):
        if self.device.type == "cuda":
            self.core.torch.cuda.synchronize(self.device)

    def read_patch(self, window, stage):
        import numpy as np
        from rasterio.windows import Window as RasterWindow

        started = time.perf_counter()
        rw = RasterWindow(window.x0, window.y0, window.size, window.size)
        patch = self.src.read([1, 2, 3], window=rw, boundless=True, fill_value=0)
        masks = self.src.read_masks([1, 2, 3], window=rw, boundless=True)
        # read_masks 的 boundless 区域以及任一 RGB 波段的 nodata 均不可见。
        valid = np.all(masks > 0, axis=0) & ~np.all(patch <= self.black_threshold, axis=0)
        patch[:, ~valid] = 0
        visible_width = max(0, min(self.src.width, window.x0 + window.size) - max(0, window.x0))
        visible_height = max(0, min(self.src.height, window.y0 + window.size) - max(0, window.y0))
        # 影像小于基础窗口时，外部填充不应使整幅有效影像被误判为空窗。
        valid_ratio = float(valid.sum()) / max(1, visible_width * visible_height)
        self.stats[stage + "_attempted_windows"] += 1
        skip = valid_ratio == 0.0 or valid_ratio < self.min_valid_ratio
        if skip:
            self.stats[stage + "_skipped_windows"] += 1
        self.window_diagnostics.append({
            "window": asdict(window), "window_id": window.window_id, "stage": stage,
            "valid_ratio": valid_ratio, "valid_ratio_full_window": float(valid.mean()), "skipped": skip,
        })
        self.stats["read_seconds"] += time.perf_counter() - started
        return None if skip else (patch, valid)

    def infer_batch(self, windows, patches, masks, stage):
        import numpy as np

        if not windows:
            return []
        if len({w.size for w in windows}) != 1:
            raise ValueError("同一 batch 的源窗口尺寸必须一致。")
        torch = self.core.torch
        self.synchronize()
        started = time.perf_counter()
        tensor = self.core.patches_to_tensor(patches, self.device)
        sizes = torch.tensor([[w.size, w.size] for w in windows], dtype=torch.int64, device=self.device)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type, enabled=self.core.USE_AMP and self.device.type == "cuda"
        ):
            output = self.model(tensor, sizes)
        labels, boxes, scores = self.core.decode_model_output(output)
        labels, boxes, scores = [v.detach().cpu().numpy() for v in (labels, boxes, scores)]
        self.synchronize()
        self.stats["model_seconds"] += time.perf_counter() - started
        self.stats["model_forward_calls"] += 1
        self.stats["model_view_count"] += len(windows)
        self.stats[stage + "_model_views"] += len(windows)
        observations = []
        for index, window in enumerate(windows):
            retained = []
            for label, box, score in zip(labels[index], boxes[index], scores[index]):
                if not np.isfinite(score) or not np.isfinite(label) or not np.all(np.isfinite(box)):
                    self.stats["discarded_invalid_predictions"] += 1
                    continue
                if not 0 <= float(score) <= 1 or float(label) != 0.0:
                    self.stats["discarded_invalid_predictions"] += 1
                    continue
                if float(score) < self.config.candidate_confidence:
                    continue
                # 先裁剪到实际可见源窗口，再裁剪到全图；不把模型越界回归当作已见上下文。
                local = np.asarray(box, dtype=float).copy()
                if local[2] <= local[0] or local[3] <= local[1]:
                    self.stats["discarded_invalid_predictions"] += 1
                    continue
                clipped = np.clip(local, 0.0, float(window.size))
                global_box = clipped + [window.x0, window.y0, window.x0, window.y0]
                global_box[[0, 2]] = np.clip(global_box[[0, 2]], 0, self.src.width)
                global_box[[1, 3]] = np.clip(global_box[[1, 3]], 0, self.src.height)
                if global_box[2] - global_box[0] < 1 or global_box[3] - global_box[1] < 1:
                    self.stats["discarded_invalid_predictions"] += 1
                    continue
                if not np.array_equal(local + [window.x0, window.y0, window.x0, window.y0], global_box):
                    self.stats["clipped_predictions"] += 1
                cx = min(window.size - 1, max(0, int((global_box[0] + global_box[2]) / 2 - window.x0)))
                cy = min(window.size - 1, max(0, int((global_box[1] + global_box[3]) / 2 - window.y0)))
                if not masks[index][cy, cx]:
                    self.stats["discarded_on_nodata_center"] += 1
                    continue
                retained.append(Observation(tuple(float(v) for v in global_box), float(score), window,
                                            class_id=1, stage=stage))
            retained.sort(key=lambda observation: observation.score, reverse=True)
            self.stats["discarded_by_topk"] += max(0, len(retained) - self.topk)
            observations.extend(retained[:self.topk])
        return observations

    def run_base(self, windows, progress=print):
        observations, pending_windows, patches, masks = [], [], [], []
        last_message = time.perf_counter()
        for index, window in enumerate(windows):
            data = self.read_patch(window, "base")
            if data is not None:
                pending_windows.append(window)
                patches.append(data[0])
                masks.append(data[1])
            if len(pending_windows) >= self.batch_size or index == len(windows) - 1:
                observations.extend(self.infer_batch(pending_windows, patches, masks, "base"))
                pending_windows, patches, masks = [], [], []
            now = time.perf_counter()
            if progress and (now - last_message >= 10 or index == len(windows) - 1):
                progress(f"[初检] {index + 1}/{len(windows)} 窗口，{len(observations)} 个候选")
                last_message = now
        return observations

    def __call__(self, window):
        data = self.read_patch(window, "refine")
        return [] if data is None else self.infer_batch([window], [data[0]], [data[1]], "refine")


def prediction_feature(prediction, transform, crs):
    from rasterio.warp import transform_geom

    x1, y1, x2, y2 = prediction.box
    # 沿仿射变换后的真实四角构造多边形，兼容旋转影像。
    corners = [list(transform * (x, y)) for x, y in ((x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1))]
    geometry = transform_geom(crs, "EPSG:4326", {"type": "Polygon", "coordinates": [corners]}, precision=9)
    polygons = [geometry["coordinates"]] if geometry["type"] == "Polygon" else geometry["coordinates"]
    for rings in polygons:
        for index, ring in enumerate(rings):
            signed_area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring, ring[1:]))
            if (signed_area < 0) == (index == 0):
                ring.reverse()  # RFC7946: 外环逆时针，内环顺时针。
    properties = asdict(prediction)
    properties["pixel_box"] = properties.pop("box")
    return {"type": "Feature", "geometry": geometry, "properties": properties}


def export_results(output_dir, result, final_predictions, src, backend, *, write_shp=False,
                   shp_filename="predictions.shp", write_gpkg=False, preview=True):
    output_dir = Path(output_dir)
    if src.crs is None:
        raise ValueError("影像缺少 CRS，禁止把原图坐标伪装成 WGS84 GeoJSON。")
    write_json(output_dir / "candidates.json", {
        "coordinate_space": "original_raster_pixel_xyxy", "width": src.width, "height": src.height,
        "predictions": [asdict(item) for item in result.predictions],
        "observations": [asdict(item) for item in result.observations],
    })
    write_json(output_dir / "targets.json", result.targets)
    with (output_dir / "events.jsonl").open("w", encoding="utf-8") as stream:
        for event in result.events:
            stream.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
    write_json(output_dir / "windows.json", backend.window_diagnostics)
    features = [prediction_feature(item, src.transform, src.crs) for item in final_predictions]
    write_json(output_dir / "predictions.geojson", {"type": "FeatureCollection", "features": features})
    fields = ["target_id", "class_id", "score", "pixel_x1", "pixel_y1", "pixel_x2", "pixel_y2",
              "status", "support_count", "refine_count", "anchor_window", "trusted_sides"]
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for item in final_predictions:
            record = asdict(item)
            box = record.pop("box")
            record.update(dict(zip(fields[3:7], box)))
            record["trusted_sides"] = json.dumps(record["trusted_sides"])
            writer.writerow(record)
    if preview:
        detections = [backend.core.Detection(*item.box, score=item.score, class_id=item.class_id)
                      for item in final_predictions]
        backend.core.save_preview(src, output_dir / "preview.png", detections, [])
    if write_shp or write_gpkg:
        _write_optional_vectors(output_dir, final_predictions, src, write_shp, write_gpkg,
                                shp_filename=shp_filename)


def _write_optional_vectors(output_dir, predictions, src, write_shp, write_gpkg, *,
                            shp_filename="predictions.shp"):
    import geopandas as gpd
    from shapely.geometry import Polygon

    records, geometries = [], []
    for item in predictions:
        x1, y1, x2, y2 = item.box
        geometries.append(Polygon([src.transform * point for point in
                                   ((x1, y1), (x1, y2), (x2, y2), (x2, y1))]))
        records.append({"target_id": item.target_id, "score": item.score, "class_id": item.class_id,
                        "status": item.status, "support": item.support_count, "refines": item.refine_count,
                        "anchor": item.anchor_window, "trusted": "".join(str(int(v)) for v in item.trusted_sides)})
    columns = ["target_id", "score", "class_id", "status", "support", "refines", "anchor", "trusted"]
    frame = gpd.GeoDataFrame(records, columns=columns, geometry=geometries, crs=src.crs)
    if write_shp:
        frame.to_file(output_dir / shp_filename, driver="ESRI Shapefile", encoding="utf-8")
    if write_gpkg:
        frame.to_file(output_dir / "predictions.gpkg", driver="GPKG", layer="predictions")


def load_ground_truths(path, src):
    """只在全部推理结束后调用。GeoJSON 无 crs 时遵循 RFC7946 的 WGS84。"""
    from rasterio.warp import transform_geom

    path = Path(path)
    if path.suffix.lower() in {".json", ".geojson"}:
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if payload.get("coordinate_space") == "pixel":
            return _validate_ground_truths(payload["ground_truths"], src)
        if payload.get("type") != "FeatureCollection":
            raise ValueError("JSON 真值需为 GeoJSON FeatureCollection 或明确 coordinate_space='pixel'。")
        source_crs = "EPSG:4326"
        if "crs" in payload:
            declared = payload["crs"]
            if declared.get("type") != "name" or not declared.get("properties", {}).get("name"):
                raise ValueError("无法识别真值 GeoJSON 的 CRS。")
            source_crs = declared["properties"]["name"]
        features = payload["features"]
    else:
        import geopandas as gpd
        frame = gpd.read_file(path)
        if frame.crs is None:
            raise ValueError("真值矢量缺少 CRS。")
        source_crs = frame.crs
        features = list(frame.iterfeatures())
    records = []
    inverse = ~src.transform
    for index, feature in enumerate(features):
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("检测真值必须是 Polygon/MultiPolygon；不支持点、线或集合。")
        geometry = transform_geom(source_crs, src.crs, geometry)
        coordinates = list(_coordinates(geometry["coordinates"]))
        if not coordinates:
            continue
        pixels = [inverse * point for point in coordinates]
        xs, ys = zip(*pixels)
        properties = feature.get("properties") or {}
        records.append({"box": [min(xs), min(ys), max(xs), max(ys)],
                        "class_id": int(properties.get("class_id", 1)),
                        "plant_id": str(properties.get("plant_id", properties.get("id", f"GT{index + 1:04d}")))})
    return _validate_ground_truths(records, src)


def _coordinates(value):
    if isinstance(value, (list, tuple)) and len(value) >= 2 and isinstance(value[0], (int, float)):
        yield float(value[0]), float(value[1])
    else:
        for child in value:
            yield from _coordinates(child)


def _validate_ground_truths(records, src):
    result = []
    identifiers = set()
    for index, record in enumerate(records):
        box = [float(value) for value in record["box"]]
        if len(box) != 4 or not all(math.isfinite(value) for value in box):
            raise ValueError("真值包含无效 xyxy 坐标。")
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError("真值框面积必须大于零。")
        class_id = int(record.get("class_id", 1))
        if class_id != 1:
            raise ValueError("当前火电厂评价仅支持 class_id=1。")
        box = [max(0, min(src.width, box[0])), max(0, min(src.height, box[1])),
               max(0, min(src.width, box[2])), max(0, min(src.height, box[3]))]
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        plant_id = str(record.get("plant_id", f"GT{index + 1:04d}"))
        if plant_id in identifiers:
            raise ValueError(f"真值 plant_id 重复: {plant_id}")
        identifiers.add(plant_id)
        result.append({"box": box, "class_id": class_id, "plant_id": plant_id})
    return result
