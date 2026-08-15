"""Shared experiment settings used by training, validation, and inference.

For ordinary S/M comparison experiments, model structure and official
pretrained weights follow ``MODEL_SIZE``. Dataset and result paths remain
explicitly editable in train.py, valid.py, and the formal inference script.
"""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

# Supported values: "s" and "m". Model YAML and official pretrained weights
# follow this selection; experiment input/output paths do not.
MODEL_SIZE = "m"
MODEL_IMAGE_SIZE = 512
# The custom VRAC YAML currently exists only for D-FINE-S. Keep this True for
# the S innovation experiment. Switching MODEL_SIZE to "m" automatically uses
# the ordinary M config instead of an incompatible VRAC config.
ENABLE_VRAC_FOR_S = False
USE_VRAC_AUGMENTATION = ENABLE_VRAC_FOR_S and MODEL_SIZE.lower() == "s"

# Network input size after preprocessing. Large-raster tile sizes may differ;
# every tile is resized to this size before it is passed to D-FINE.


MODEL_CONFIG_PATHS = {
    "s": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_s_custom.yml",
    "m": REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_m_custom.yml",
}
VRAC_S_CONFIG_PATH = (
    REPO_ROOT / "configs" / "dfine" / "custom" / "dfine_hgnetv2_s_custom_vrac.yml"
)
PRETRAINED_WEIGHT_PATHS = {
    "s": REPO_ROOT / "weight" / "dfine_s_obj2coco.pth",
    "m": REPO_ROOT / "weight" / "dfine_m_obj2coco.pth",
}


def normalized_model_size() -> str:
    """Return and validate the shared D-FINE model size."""
    model_size = MODEL_SIZE.lower().strip()
    if model_size not in MODEL_CONFIG_PATHS:
        raise ValueError(f'MODEL_SIZE must be "s" or "m", got {MODEL_SIZE!r}')
    return model_size


def selected_model_config_path() -> Path:
    """Return the model YAML selected by MODEL_SIZE and the S-only VRAC flag."""
    model_size = normalized_model_size()
    return VRAC_S_CONFIG_PATH if USE_VRAC_AUGMENTATION else MODEL_CONFIG_PATHS[model_size]


MODEL_TAG = f"D-FINE-{normalized_model_size().upper()}" + (
    "_VRAC" if USE_VRAC_AUGMENTATION else ""
)
PRETRAINED_WEIGHT_PATH = PRETRAINED_WEIGHT_PATHS[normalized_model_size()]
