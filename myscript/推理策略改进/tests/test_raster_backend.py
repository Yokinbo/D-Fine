"""真实栅格读写与入口集成测试；检测器使用替身，不读取训练权重。

在 dfine 环境运行；缺少 torch/numpy/rasterio 时自动跳过本文件。
"""

from contextlib import redirect_stdout
from dataclasses import replace
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

SCRIPT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPT_DIR))

try:
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    import torch
    HAS_RUNTIME = True
except ImportError:
    HAS_RUNTIME = False

import raster_backend
from strategy import Prediction, StrategyConfig, Window


class FakeDetector:
    def __init__(self, boxes, scores=None, labels=None):
        self.boxes = boxes
        self.scores = scores if scores is not None else [0.8] * len(boxes)
        self.labels = labels if labels is not None else [0] * len(boxes)
        self.calls = 0
        self.input_shapes = []

    def __call__(self, tensor, sizes):
        self.calls += 1
        self.input_shapes.append(tuple(tensor.shape))
        batch = len(sizes)
        return (torch.tensor([self.labels] * batch, device=tensor.device),
                torch.tensor([self.boxes] * batch, dtype=torch.float32, device=tensor.device),
                torch.tensor([self.scores] * batch, dtype=torch.float32, device=tensor.device))


@unittest.skipUnless(HAS_RUNTIME, "需要 dfine 环境的 torch/numpy/rasterio")
class RasterIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("crop_response_cli_tests", SCRIPT_DIR / "裁剪响应自适应推理.py")
        cls.cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.cli)
        cls.args = cls.cli.build_parser().parse_args(["--device", "cpu", "--no-amp"])
        cls.config = replace(StrategyConfig(), base_size=64, stride=32, scales=(64, 128),
                             model_input_size=32, shift_pixels=8)
        # 使用仓库真实的张量预处理与输出解码函数，模型前向由替身提供。
        cls.core = raster_backend.load_inference_core(cls.args, cls.config)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="crop_response_test_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def raster(self, *, width=96, height=80, black=False):
        filename = self.root / "image.tif"
        data = np.full((3, height, width), 0 if black else 128, dtype="uint8")
        with rasterio.open(filename, "w", driver="GTiff", width=width, height=height,
                           count=3, dtype="uint8", crs="EPSG:3857", nodata=0,
                           transform=from_origin(12000000, 4000000, 2, 2)) as dst:
            dst.write(data)
        return filename

    def backend(self, src, detector, **kwargs):
        return raster_backend.RasterBackend(src, detector, torch.device("cpu"), self.core,
                                            self.config, **kwargs)

    def test_padded_small_image_uses_visible_area_and_clips_boxes(self):
        detector = FakeDetector([[-5, -5, 70, 70]])
        with rasterio.open(self.raster(width=32, height=32)) as src:
            backend = self.backend(src, detector)
            observations = backend.run_base([Window(0, 0, 64)], progress=None)
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].box, (0, 0, 32, 32))
        self.assertEqual(backend.window_diagnostics[0]["valid_ratio"], 1.0)
        self.assertEqual(backend.stats["clipped_predictions"], 1)
        self.assertEqual(detector.input_shapes, [(1, 3, 32, 32)])

    def test_nodata_window_never_calls_detector(self):
        detector = FakeDetector([[6, 7, 24, 25]])
        with rasterio.open(self.raster(black=True)) as src:
            backend = self.backend(src, detector)
            self.assertEqual(backend.run_base([Window(0, 0, 64)], progress=None), [])
        self.assertEqual(detector.calls, 0)
        self.assertEqual(backend.stats["base_skipped_windows"], 1)
        self.assertEqual(backend.stats["model_view_count"], 0)

    def test_batch_mapping_and_invalid_query_filter(self):
        detector = FakeDetector([[4, 5, 20, 22], [float("nan"), 1, 20, 22],
                                 [30, 30, 29, 40], [4, 5, 20, 22]],
                                scores=[0.8, 0.9, 0.9, 0.9], labels=[0, 0, 0, 1])
        with rasterio.open(self.raster()) as src:
            backend = self.backend(src, detector, batch_size=2)
            observations = backend.run_base([Window(0, 0, 64), Window(32, 16, 64)], progress=None)
        self.assertEqual([o.box for o in observations], [(4, 5, 20, 22), (36, 21, 52, 38)])
        self.assertEqual(backend.stats["model_forward_calls"], 1)
        self.assertEqual(backend.stats["model_view_count"], 2)
        self.assertEqual(backend.stats["discarded_invalid_predictions"], 6)

    def test_geojson_roundtrip_recovers_original_pixel_box(self):
        prediction = Prediction((10, 12, 50, 55), 0.8)
        with rasterio.open(self.raster()) as src:
            feature = raster_backend.prediction_feature(prediction, src.transform, src.crs)
            ring = feature["geometry"]["coordinates"][0]
            self.assertTrue(all(-180 <= x <= 180 and -90 <= y <= 90 for x, y in ring))
            geojson = self.root / "truth.geojson"
            raster_backend.write_json(geojson, {"type": "FeatureCollection", "features": [feature]})
            truths = raster_backend.load_ground_truths(geojson, src)
        self.assertEqual(len(truths), 1)
        for actual, expected in zip(truths[0]["box"], prediction.box):
            self.assertAlmostEqual(actual, expected, places=3)

    def test_full_entry_exports_and_only_then_evaluates(self):
        filename = self.raster(width=32, height=32)
        checkpoint = self.root / "fake.pth"
        checkpoint.write_bytes(b"fake detector; never deserialize as a checkpoint")
        ground_truth = self.root / "gt.json"
        raster_backend.write_json(ground_truth, {"coordinate_space": "pixel", "ground_truths": [
            {"box": [6, 7, 24, 25], "plant_id": "plant1", "class_id": 1}]})
        output = self.root / "results"
        argv = ["--input", str(filename), "--checkpoint", str(checkpoint), "--output", str(output),
                "--gt", str(ground_truth), "--device", "cpu", "--no-amp", "--no-preview",
                "--strategy", "overlap", "--base-size", "64", "--stride", "32",
                "--scales", "64", "128", "--model-input-size", "32", "--shift-pixels", "8"]
        detector = FakeDetector([[6, 7, 24, 25]])
        original_load_gt = raster_backend.load_ground_truths

        def load_after_inference(path, src):
            self.assertEqual(detector.calls, 1)
            self.assertTrue((output / "predictions.geojson").is_file())
            return original_load_gt(path, src)

        with patch.object(raster_backend, "load_inference_core", return_value=self.core), \
                patch.object(self.core, "load_model", return_value=detector), \
                patch.object(raster_backend, "load_ground_truths", side_effect=load_after_inference), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(self.cli.main(argv), 0)
        metadata = json.loads((output / "run.json").read_text(encoding="utf-8"))
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        self.assertEqual(metadata["status"], "completed")
        self.assertTrue(metadata["ground_truth_used_after_inference"])
        self.assertFalse(metadata["model_settings_switch_used"])
        self.assertEqual(metadata["backend_stats"]["model_view_count"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["scene_single_class_AP50_101pt"], 1.0)
        self.assertIsNone(metrics["fp_per_km2"])
        for name in ("candidates.json", "events.jsonl", "targets.json", "windows.json",
                     "predictions.csv", "predictions.geojson", "effective_model_config.json"):
            self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
