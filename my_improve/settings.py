"""论文改进实验的统一开关。

修改 ``IMPROVEMENT_MODE`` 后，train.py、valid.py 和 test.py 会经由
``experiment_config.py`` 同步选用相同网络配置，避免训练与评估结构不一致。
"""

from pathlib import Path


IMPROVEMENT_DIR = Path(__file__).resolve().parent

# QFBCG 模式保留用于复现已经完成的失败实验；后续第二模块改用 DSQC。
# "baseline"：原始 D-FINE-M
# "qlcs"：仅开启查询引导的隐式部件采样
# "dsqc"：仅开启解码稳定性感知查询校准
# "qlcs_dsqc"：同时开启 QLCS 和 DSQC（新的第二步累计消融）
# "qacg"：仅开启质量感知查询竞争门控
# "qlcs_dsqc_qacg"：依次开启 QLCS、DSQC 和 QACG（第三步累计消融）
# "qfbcg" / "qlcs_qfbcg"：旧 QFBCG 复现实验，不建议继续作为正式方案
# "qlcs_dsqc_csga": QLCS + DSQC + cross-scale guided alignment (experimental).    待训练
# "dsqc_csga": DSQC + CSGA, QLCS disabled; next validation ablation.
IMPROVEMENT_MODE = "dsqc_csga"

IMPROVEMENT_CONFIG_PATHS = {
    "dsqc_csga": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_dsqc_csga.yml",
    },
    "qlcs_dsqc_csga": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qlcs_dsqc_csga.yml",
    },
    "qlcs_dsqc_qcr": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qlcs_dsqc_qcr.yml",
    },
    "qlcs_dsqc_shea": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qlcs_dsqc_shea.yml",
    },
    "mgca": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_mgca.yml",
    },
    "qlcs_dsqc_mgca": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qlcs_dsqc_mgca.yml",
    },
    "qlcs": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qlcs.yml",
    },
    "qfbcg": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qfbcg.yml",
    },
    "qlcs_qfbcg": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qlcs_qfbcg.yml",
    },
    "dsqc": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_dsqc.yml",
    },
    "qlcs_dsqc": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qlcs_dsqc.yml",
    },
    "qacg": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qacg.yml",
    },
    "qlcs_dsqc_qacg": {
        "m": IMPROVEMENT_DIR / "dfine_hgnetv2_m_qlcs_dsqc_qacg.yml",
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
            "网络结构消融期间请关闭 VRAC，避免增强增益与网络模块贡献混淆。"
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
