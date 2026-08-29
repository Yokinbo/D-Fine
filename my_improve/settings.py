"""论文改进实验的统一开关。

修改 ``IMPROVEMENT_MODE`` 后，train.py、valid.py 和 test.py 会经由
``experiment_config.py`` 同步选用相同网络配置，避免训练与评估结构不一致。
"""

from pathlib import Path


IMPROVEMENT_DIR = Path(__file__).resolve().parent

# 当前第一步只开放 QLCS 单模块消融。
# 可选值："baseline"（原始 D-FINE）或 "qlcs"（D-FINE + QLCS）。
IMPROVEMENT_MODE = "baseline"

IMPROVEMENT_CONFIG_PATHS = {
    "qlcs": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qlcs.yml",
    },
}


def normalized_improvement_mode() -> str:
    """规范化并检查当前改进模式。"""
    mode = IMPROVEMENT_MODE.lower().strip()
    supported = {"baseline", *IMPROVEMENT_CONFIG_PATHS}
    if mode not in supported:
        choices = "、".join(sorted(supported))
        raise ValueError(f"IMPROVEMENT_MODE 必须是 {choices}，当前为 {IMPROVEMENT_MODE!r}")
    return mode


def selected_improvement_config_path(
    model_size: str,
    baseline_config_path: Path,
    use_vrac_augmentation: bool,
) -> Path:
    """返回改进 YAML；baseline 模式原样返回既有配置。"""
    mode = normalized_improvement_mode()
    if mode == "baseline":
        return baseline_config_path
    if use_vrac_augmentation:
        raise ValueError(
            "第一阶段应单独验证网络改进，请先关闭 VRAC；待 QLCS 单模块消融完成后再做组合实验。"
        )

    configs = IMPROVEMENT_CONFIG_PATHS[mode]
    if model_size not in configs:
        supported_sizes = "、".join(size.upper() for size in sorted(configs))
        raise ValueError(f"{mode.upper()} 当前仅提供 {supported_sizes} 规模配置")
    return configs[model_size]


def improvement_tag() -> str:
    """返回用于模型显示名称的改进后缀。"""
    mode = normalized_improvement_mode()
    return "" if mode == "baseline" else f"_{mode.upper()}"
