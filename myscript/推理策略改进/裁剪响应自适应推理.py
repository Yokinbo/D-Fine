"""裁剪边界响应引导的火电厂目标完整性恢复推理。

修改用户配置区即可运行；--help / --dry-run 不需要 torch 或 GIS 依赖。
所有创新策略都位于本目录；网络使用显式 DSQC+RBA 配置和用户指定权重。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]

# =============================================================================
# 用户配置区：先填写实际 DSQC+RBA 权重，输入为与训练辐射范围一致的 uint8 RGB。
# MODEL_CONFIG 明确指定 DSQC+RBA，不跟随 my_improve/settings.py 当前开关。
# =============================================================================
INPUT_TIF = r"G:\金三角tif影像\宁夏\银川市\灵武市1.88m\Level16\灵武市1.88m.tif"
CHECKPOINT = r"G:\b1完整目标检测模型与权重结果\权重结果\改进实验dfine\dsqc_rba\最佳2e-4_SD18\best_map50.pth"  # 必填：经过验证的 DSQC+RBA 检测权重 .pth。
MODEL_CONFIG = str(REPO_ROOT / "my_improve" / "dfine_hgnetv2_m_dsqc_rba.yml")
OUTPUT_DIR = r"F:\3能源金三角基础设施识别\火力发电厂\火电厂论文撰写\大图推理策略实验\灵武推理\1response0.8"  # 留空则自动新建 本目录/results/时间戳，防止覆盖其他实验。
GROUND_TRUTH = ""  # 可选 SHP/GPKG/GeoJSON；仅在全部推理结束之后读取。
STRATEGY = "response"  # response / static / fixed / random / overlap
WRITE_SHP = True  # True：输出 SHP；False：不输出 SHP。直接改这里，无需命令行参数。
SHP_FILENAME = "response0.8灵武推理.shp"  # 可自定义，例如 "准格尔召镇检测结果.shp"；仅填文件名且保留 .shp。
DEVICE = "cuda:0"
BATCH_SIZE = 6
MODEL_INPUT_SIZE = 512  # 与本仓库 experiment_config.MODEL_IMAGE_SIZE=512 保持一致。

FINAL_CONFIDENCE = 0.8  #推理置信度

BASE_SIZE = 512
STRIDE = 256
SCALES = (512, 768, 1024)
MAX_REFINES_PER_TARGET = 3
GLOBAL_REFINE_BUDGET = 1000
# 其余算法阈值见 strategy.StrategyConfig；可用 --strategy-config JSON 文件覆盖。
# 所有默认阈值仅为第一版验证起点，正式测试前在独立验证区域固定。
# =============================================================================


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=INPUT_TIF, help="至少三个 uint8 RGB 波段且具 CRS 的 GeoTIFF")
    parser.add_argument("--checkpoint", default=CHECKPOINT, help="必填 DSQC+RBA 权重，严格检查参数匹配")
    parser.add_argument("--config", default=MODEL_CONFIG, help="显式模型 YAML，不读取当前改进开关")
    parser.add_argument("--output", default=OUTPUT_DIR, help="新建输出目录；默认按时间戳创建")
    parser.add_argument("--strategy", choices=["response", "static", "fixed", "random", "overlap"], default=STRATEGY)
    parser.add_argument("--strategy-config", default="", help="算法阈值 JSON；命令行显式参数优先")
    parser.add_argument("--gt", default=GROUND_TRUTH, help="可选真值路径，仅推理结束后读取")
    parser.add_argument("--confidence", type=float, default=FINAL_CONFIDENCE, help="最终导出及P/R/F1阈值；AP仍使用低阈值候选")
    parser.add_argument("--candidate-confidence", type=float, default=0.05)
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model-input-size", type=int, default=MODEL_INPUT_SIZE)
    parser.add_argument("--base-size", type=int, default=BASE_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--scales", type=int, nargs="+", default=SCALES)
    parser.add_argument("--shift-pixels", type=int, default=64)
    parser.add_argument("--max-refines", type=int, default=MAX_REFINES_PER_TARGET)
    parser.add_argument("--global-budget", type=int, default=GLOBAL_REFINE_BUDGET)
    parser.add_argument("--grid-offset", type=int, nargs=2, default=(0, 0), metavar=("X", "Y"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-side-fusion", action="store_true")
    parser.add_argument("--topk", type=int, default=50, help="每窗口低阈值候选上限，所有策略统一")
    parser.add_argument("--black-threshold", type=int, default=3)
    parser.add_argument("--min-valid-ratio", type=float, default=0.2)
    parser.add_argument("--valid-area-km2", type=float, default=None, help="独立提供测试区有效面积；不自动把角度转换成平方米")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--preview-max-size", type=int, default=2400)
    parser.add_argument("--write-shp", dest="write_shp", action="store_true", default=WRITE_SHP,
                        help="保存原影像 CRS Shapefile；平时直接修改顶部 WRITE_SHP")
    parser.add_argument("--shp-filename", default=SHP_FILENAME,
                        help="SHP 文件名；平时直接修改顶部 SHP_FILENAME")
    parser.add_argument("--write-gpkg", action="store_true", help="额外保存原影像 CRS GeoPackage，需 geopandas")
    parser.add_argument("--dry-run", action="store_true", help="只显示参数和路径检查，不导入模型、读取影像或创建结果")
    return parser


def configuration(args, argv):
    from strategy import StrategyConfig

    values = asdict(StrategyConfig())
    if args.strategy_config:
        overrides = json.loads(Path(args.strategy_config).read_text(encoding="utf-8-sig"))
        unknown = set(overrides) - set(values)
        if unknown:
            raise ValueError(f"算法 JSON 存在未知字段: {sorted(unknown)}")
        values.update(overrides)
    options = {
        "--strategy": ("policy", args.strategy), "--candidate-confidence": ("candidate_confidence", args.candidate_confidence),
        "--model-input-size": ("model_input_size", args.model_input_size), "--base-size": ("base_size", args.base_size),
        "--stride": ("stride", args.stride), "--scales": ("scales", tuple(args.scales)),
        "--shift-pixels": ("shift_pixels", args.shift_pixels), "--max-refines": ("max_refines_per_target", args.max_refines),
        "--global-budget": ("max_refine_windows", args.global_budget), "--grid-offset": ("grid_offset", tuple(args.grid_offset)),
        "--seed": ("random_seed", args.seed), "--no-side-fusion": ("use_side_fusion", not args.no_side_fusion),
    }
    supplied = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    for option, (field, value) in options.items():
        if not args.strategy_config or option in supplied:
            values[field] = value
    values["scales"], values["grid_offset"] = tuple(values["scales"]), tuple(values["grid_offset"])
    config = StrategyConfig(**values)
    config.validate()
    if not math.isfinite(args.confidence) or not config.candidate_confidence <= args.confidence <= 1:
        raise ValueError("最终置信度必须介于候选阈值和1之间。")
    if args.batch_size <= 0 or args.topk <= 0 or args.preview_max_size <= 0:
        raise ValueError("batch-size/topk/preview-max-size 必须为正数。")
    if not 0 <= args.black_threshold <= 255 or not math.isfinite(args.min_valid_ratio) or not 0 <= args.min_valid_ratio <= 1:
        raise ValueError("black-threshold 应在0~255；min-valid-ratio 应在0~1。")
    if args.valid_area_km2 is not None and (not math.isfinite(args.valid_area_km2) or args.valid_area_km2 <= 0):
        raise ValueError("valid-area-km2 必须为有限正数。")
    if args.write_shp:
        shp_filename = args.shp_filename.strip()
        invalid_chars = set('<>:"/\\|?*')
        if (not shp_filename or Path(shp_filename).name != shp_filename
                or Path(shp_filename).suffix.lower() != ".shp"
                or any(char in invalid_chars for char in shp_filename)):
            raise ValueError("SHP_FILENAME 只能填写以 .shp 结尾的文件名，不能包含目录或 Windows 非法字符。")
        args.shp_filename = shp_filename
    return config


def run(args, config):
    from raster_backend import (RasterBackend, export_results, file_sha256, load_ground_truths,
                                load_inference_core, write_json)
    from strategy import grid_windows, run_strategy

    required = {"输入影像": args.input, "模型配置": args.config, "权重": args.checkpoint}
    missing = [f"{key}: {value or '(未填写)'}" for key, value in required.items() if not value or not Path(value).is_file()]
    if missing:
        raise FileNotFoundError("请填写有效路径：\n" + "\n".join(missing))
    output = Path(args.output) if args.output else SCRIPT_DIR / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"输出目录非空，为保留可复现实验结果，请指定新目录: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    metadata = {
        "status": "running", "started_at": datetime.now().astimezone().isoformat(),
        "output": str(output), "arguments": vars(args), "strategy_config": asdict(config),
        "input": str(Path(args.input).resolve()), "checkpoint": str(Path(args.checkpoint).resolve()),
        "model_config": str(Path(args.config).resolve()), "model_input_size": config.model_input_size,
        "strict_state_dict_loading": True, "model_settings_switch_used": False,
        "valid_area_km2": args.valid_area_km2,
        "area_source": "user_supplied" if args.valid_area_km2 is not None else "unavailable",
        "candidate_filter": "finite, class 1, clip to visible crop and raster, valid center, confidence, topk",
        "evaluation_reference_offset": [0, 0],
        "timing_note": "inference_seconds 包含滑窗读图/前向/策略；不含权重加载、预热、输出及真值评价。",
    }
    write_json(output / "run.json", metadata)
    try:
        print(f"[配置] 策略={config.policy} | 模型={args.config} | 输入尺寸={config.model_input_size}", flush=True)
        print(f"[输出] {output}", flush=True)
        print("[模型] 计算权重摘要并加载模型（严格参数匹配）...", flush=True)
        metadata["checkpoint_sha256"] = file_sha256(args.checkpoint)
        metadata["model_config_sha256"] = file_sha256(args.config)
        metadata["implementation_sha256"] = {
            name: file_sha256(SCRIPT_DIR / name) for name in
            ("strategy.py", "raster_backend.py", "裁剪响应自适应推理.py", "evaluation.py")
            if (SCRIPT_DIR / name).is_file()
        }
        core = load_inference_core(args, config)
        core.torch.manual_seed(config.random_seed)
        core.np.random.seed(config.random_seed)
        device = core.resolve_device()
        metadata["runtime"] = {"python": sys.version, "torch": str(core.torch.__version__),
                               "rasterio": str(core.rasterio.__version__), "numpy": str(core.np.__version__),
                               "device": str(device), "amp_enabled": not args.no_amp and device.type == "cuda"}
        # 保存含 include 展开的有效 YAML，方便核对模型开关和输入尺寸。
        effective = core.YAMLConfig(str(Path(args.config).resolve()), num_classes=1,
                                   remap_mscoco_category=False,
                                   eval_spatial_size=[config.model_input_size, config.model_input_size])
        if "HGNetv2" in effective.yaml_cfg:
            effective.yaml_cfg["HGNetv2"]["pretrained"] = False
        write_json(output / "effective_model_config.json", effective.yaml_cfg)
        with core.rasterio.open(args.input) as src:
            # 先检查影像、空间参考，再花时间加载模型。
            backend = RasterBackend(src, None, device, core, config, batch_size=args.batch_size,
                                    topk=args.topk, black_threshold=args.black_threshold,
                                    min_valid_ratio=args.min_valid_ratio)
            metadata["raster"] = {"width": src.width, "height": src.height, "bands": src.count,
                                  "dtypes": src.dtypes, "crs": str(src.crs),
                                  "transform": list(src.transform), "nodata": src.nodata}
            load_started = time.perf_counter()
            backend.model = core.load_model(device)
            metadata["model_load_and_warmup_seconds"] = time.perf_counter() - load_started
            backend.synchronize()
            infer_started = time.perf_counter()
            windows = grid_windows(src.width, src.height, config.base_size, config.stride, config.grid_offset)
            print(f"[初检] 共 {len(windows)} 个基础窗口", flush=True)
            observations = backend.run_base(windows)
            result = run_strategy(observations, windows, backend, src.width, src.height, config=config,
                                  progress=lambda message: print(message, flush=True))
            backend.synchronize()
            metadata["inference_seconds"] = time.perf_counter() - infer_started
            metadata["strategy_stats"], metadata["backend_stats"] = result.stats, backend.stats
            final = sorted((item for item in result.predictions if item.score >= args.confidence),
                           key=lambda item: item.score, reverse=True)
            metadata["candidate_predictions"] = len(result.predictions)
            metadata["exported_predictions"] = len(final)
            export_results(output, result, final, src, backend, write_shp=args.write_shp,
                           shp_filename=args.shp_filename,
                           write_gpkg=args.write_gpkg, preview=not args.no_preview)
            if args.gt:
                from evaluation import evaluate_predictions
                print("[评价] 所有推理已结束，现在读取真值。", flush=True)
                truths = load_ground_truths(args.gt, src)
                metrics = evaluate_predictions(result.predictions, truths, width=src.width, height=src.height,
                                               base_size=config.base_size, reference_offset=(0, 0),
                                               final_confidence=args.confidence, match_iou=0.5,
                                               valid_area_km2=args.valid_area_km2)
                write_json(output / "metrics.json", metrics)
                write_json(output / "ground_truths_pixel.json", {"coordinate_space": "pixel", "ground_truths": truths})
                metadata["ground_truth_count"] = len(truths)
                metadata["ground_truth_used_after_inference"] = True
        metadata["status"] = "completed"
        metadata["total_seconds"] = time.perf_counter() - started
        write_json(output / "run.json", metadata)
        print(f"[完成] 输出 {metadata['exported_predictions']} 个火电厂框；推理 {metadata['inference_seconds']:.2f} 秒。", flush=True)
        print(f"[结果] {output}", flush=True)
    except Exception as error:
        metadata.update(status="failed", error=f"{type(error).__name__}: {error}",
                        total_seconds=time.perf_counter() - started)
        write_json(output / "run.json", metadata)
        raise


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = configuration(args, argv)
        if args.dry_run:
            print(json.dumps({"mode": "dry_run", "strategy_config": asdict(config), "arguments": vars(args),
                              "path_exists": {name: bool(value and Path(value).is_file()) for name, value in
                                              (("input", args.input), ("checkpoint", args.checkpoint), ("config", args.config))},
                              "note": "仅检查配置；未读取影像、加载权重或创建输出。未填写权重是正式运行前的待办。"},
                             ensure_ascii=False, indent=2, allow_nan=False))
            return 0
        run(args, config)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
