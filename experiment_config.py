"""训练、验证和测试共同使用的 D-FINE 模型配置。

只需修改 ``MODEL_SIZE``，train.py、valid.py 和 test.py 就会同步切换
模型结构。训练脚本还会自动选择与该规模匹配的官方预训练权重。
数据集路径、输出目录及待评估权重路径仍在各入口脚本中单独填写。
论文网络改进的统一开关位于 ``my_improve/settings.py``。
"""

from pathlib import Path

from my_improve.settings import (
    improvement_tag,
    normalized_improvement_mode,
    selected_improvement_config_path,
)


REPO_ROOT = Path(__file__).resolve().parent

# 可选值："s"、"m"、"l"、"x"，分别表示 Small、Medium、Large、XLarge。
# 该参数控制模型 YAML 和训练预训练权重，但不会自动修改实验输出目录名。
MODEL_SIZE = "m"

# 论文对比实验统一使用的模型输入尺寸。大图切片尺寸可以不同，送入模型前
# 会被缩放到该尺寸；512 可以被 D-FINE 的网络步长整除。
MODEL_IMAGE_SIZE = 512

# 是否在训练集上启用 VRAC（可见比例约束裁剪）及配套遥感影像增强。
# 该开关对 S/M/L/X 四种规模统一生效；验证、测试和大图推理不会执行随机增强。
ENABLE_VRAC_AUGMENTATION = False
USE_VRAC_AUGMENTATION = ENABLE_VRAC_AUGMENTATION

# 当前网络改进状态，供训练、验证和测试脚本统一显示与核对。
# 状态直接由 my_improve/settings.py 生成，请勿在这里单独修改。
ACTIVE_IMPROVEMENT_MODE = normalized_improvement_mode()
USE_QLCS = ACTIVE_IMPROVEMENT_MODE in {"qlcs", "qlcs_qfbcg", "qlcs_dsqc"}
USE_QFBCG = ACTIVE_IMPROVEMENT_MODE in {"qfbcg", "qlcs_qfbcg"}
USE_DSQC = ACTIVE_IMPROVEMENT_MODE in {"dsqc", "qlcs_dsqc"}

# ENABLE_VRAC_AUGMENTATION = False：当前规模的普通 D-FINE 基线
# ENABLE_VRAC_AUGMENTATION = True ：当前规模的 D-FINE + VRAC 训练增强


MODEL_CONFIG_PATHS = {
    "s": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_s_custom.yml",
    "m": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_m_custom.yml",
    "l": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_l_custom.yml",
    "x": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_x_custom.yml",
}

# 四种规模分别继承各自的基础模型配置，再加载完全相同的 VRAC 训练增强。
VRAC_CONFIG_PATHS = {
    "s": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_s_custom_vrac.yml",
    "m": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_m_custom_vrac.yml",
    "l": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_l_custom_vrac.yml",
    "x": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_x_custom_vrac.yml",
}
PRETRAINED_WEIGHT_PATHS = {
    "s": REPO_ROOT / "weight" / "dfine_s_obj2coco.pth",
    "m": REPO_ROOT / "weight" / "dfine_m_obj2coco.pth",
    "l": REPO_ROOT / "weight" / "dfine_l_obj2coco_e25.pth",
    "x": REPO_ROOT / "weight" / "dfine_x_obj2coco.pth",
}


def normalized_model_size() -> str:
    """规范化并检查用户选择的 D-FINE 模型规模。"""
    model_size = MODEL_SIZE.lower().strip()
    if model_size not in MODEL_CONFIG_PATHS:
        raise ValueError(
            f'MODEL_SIZE 必须是 "s"、"m"、"l" 或 "x"，当前为 {MODEL_SIZE!r}'
        )
    return model_size


def selected_model_config_path() -> Path:
    """返回当前规模、增强方式及网络改进完全一致的 YAML。"""
    model_size = normalized_model_size()
    config_paths = VRAC_CONFIG_PATHS if USE_VRAC_AUGMENTATION else MODEL_CONFIG_PATHS
    return selected_improvement_config_path(
        model_size,
        config_paths[model_size],
        USE_VRAC_AUGMENTATION,
    )


MODEL_TAG = f"D-FINE-{normalized_model_size().upper()}" + (
    "_VRAC" if USE_VRAC_AUGMENTATION else ""
) + improvement_tag()
PRETRAINED_WEIGHT_PATH = PRETRAINED_WEIGHT_PATHS[normalized_model_size()]
