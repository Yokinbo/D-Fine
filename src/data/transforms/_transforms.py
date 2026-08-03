"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from typing import Any, Dict, List, Optional

import PIL
import PIL.Image
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

from ...core import register
from .._misc import (
    BoundingBoxes,
    Image,
    Mask,
    SanitizeBoundingBoxes,
    Video,
    _boxes_keys,
    convert_to_tv_tensor,
)

torchvision.disable_beta_transforms_warning()


RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
RandomVerticalFlip = register()(T.RandomVerticalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name="SanitizeBoundingBoxes")(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode="constant") -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params["padding"]
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]["padding"] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(
        self,
        min_scale: float = 0.3,
        max_scale: float = 1,
        min_aspect_ratio: float = 0.5,
        max_aspect_ratio: float = 2,
        sampler_options: Optional[List[float]] = None,
        trials: int = 40,
        p: float = 1.0,
    ):
        super().__init__(
            min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials
        )
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class RemoteSensingColorJitter(T.ColorJitter):
    """Mild RGB jitter without the channel permutation used by SSD distortion."""

    def __init__(
        self,
        brightness=0.15,
        contrast=0.2,
        saturation=0.15,
        hue=0.02,
        p: float = 0.5,
    ) -> None:
        super().__init__(brightness, contrast, saturation, hue)
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.p = float(p)

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1).item() >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]
        return super().forward(*inputs)


@register()
class RemoteSensingGaussianBlur(nn.Module):
    """Mild blur for different tile sharpness and atmospheric clarity."""

    def __init__(self, kernel_size=3, sigma=(0.1, 1.2), p: float = 0.2) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size, kernel_size]
        if any(size % 2 == 0 or size < 1 for size in kernel_size):
            raise ValueError("kernel_size values must be positive odd integers.")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.p = float(p)

    @staticmethod
    def _unpack(inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("RemoteSensingGaussianBlur expects (image, target[, dataset]).")
        return sample

    def _unchanged(self, sample):
        return tuple(sample) if isinstance(sample, list) else sample

    def forward(self, *inputs: Any) -> Any:
        sample = self._unpack(inputs)
        image, target, *extra = sample
        if torch.rand(1).item() >= self.p:
            return self._unchanged(sample)

        if isinstance(self.sigma, (tuple, list)):
            sigma = torch.empty(1).uniform_(float(self.sigma[0]), float(self.sigma[1])).item()
        else:
            sigma = float(self.sigma)
        return TF.gaussian_blur(image, self.kernel_size, [sigma, sigma]), target, *extra


@register()
class RemoteSensingGaussianNoise(nn.Module):
    """Mild additive Gaussian noise for compression and imaging noise."""

    def __init__(self, std: float = 0.02, p: float = 0.2) -> None:
        super().__init__()
        if not 0.0 <= std <= 1.0:
            raise ValueError("std must be in [0, 1].")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        self.std = float(std)
        self.p = float(p)

    @staticmethod
    def _unpack(inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("RemoteSensingGaussianNoise expects (image, target[, dataset]).")
        return sample

    def _unchanged(self, sample):
        return tuple(sample) if isinstance(sample, list) else sample

    def forward(self, *inputs: Any) -> Any:
        sample = self._unpack(inputs)
        image, target, *extra = sample
        if torch.rand(1).item() >= self.p:
            return self._unchanged(sample)

        tensor = TF.pil_to_tensor(image).float() / 255.0
        noisy = (tensor + torch.randn_like(tensor) * self.std).clamp(0.0, 1.0)
        return TF.to_pil_image(noisy), target, *extra


@register()
class VisibleRatioAwareCrop(nn.Module):
    """Randomly crop while preserving a minimum fraction of every object.

    The stock ``RandomIoUCrop`` constrains overlap between a sampled window
    and a box, but it does not directly constrain how much of a large object
    remains visible.  This transform accepts a crop only when the retained
    box area divided by the original box area is at least
    ``min_visible_ratio`` for every object in the image.

    Large objects that already touch an image boundary are protected by
    default.  Cropping them again would turn an already truncated object into
    a small, semantically incomplete fragment.
    """

    def __init__(
        self,
        min_visible_ratio: float = 0.7,
        min_scale: float = 0.7,
        max_scale: float = 1.0,
        trials: int = 40,
        p: float = 0.3,
        protect_edge_boxes: bool = True,
        edge_margin: float = 2.0,
    ) -> None:
        super().__init__()

        if not 0.0 < min_visible_ratio <= 1.0:
            raise ValueError("min_visible_ratio must be in (0, 1].")
        if not 0.0 < min_scale <= max_scale <= 1.0:
            raise ValueError(
                "min_scale and max_scale must satisfy 0 < min_scale <= max_scale <= 1."
            )
        if trials < 1:
            raise ValueError("trials must be at least 1.")
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must be in [0, 1].")
        if edge_margin < 0.0:
            raise ValueError("edge_margin must be non-negative.")

        self.min_visible_ratio = float(min_visible_ratio)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.trials = int(trials)
        self.p = float(p)
        self.protect_edge_boxes = bool(protect_edge_boxes)
        self.edge_margin = float(edge_margin)

    @staticmethod
    def _box_area(boxes: torch.Tensor) -> torch.Tensor:
        wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=0)
        return wh[:, 0] * wh[:, 1]

    @staticmethod
    def _unpack(inputs):
        sample = inputs if len(inputs) > 1 else inputs[0]
        if not isinstance(sample, (tuple, list)) or len(sample) < 2:
            raise TypeError("VisibleRatioAwareCrop expects (image, target[, dataset]).")
        return sample

    def _unchanged(self, sample):
        return tuple(sample) if isinstance(sample, list) else sample

    def forward(self, *inputs: Any) -> Any:
        sample = self._unpack(inputs)
        image, target, *extra = sample

        if torch.rand(1).item() >= self.p or "boxes" not in target:
            return self._unchanged(sample)

        boxes = target["boxes"]
        if boxes.numel() == 0:
            return self._unchanged(sample)

        if hasattr(F, "get_size"):
            image_h, image_w = F.get_size(image)
        else:  # torchvision 0.15.2
            image_h, image_w = F.get_spatial_size(image)
        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        original_area = self._box_area(boxes_tensor)
        if torch.any(original_area <= 0):
            return self._unchanged(sample)

        if self.protect_edge_boxes:
            touches_edge = (
                (boxes_tensor[:, 0] <= self.edge_margin)
                | (boxes_tensor[:, 1] <= self.edge_margin)
                | (boxes_tensor[:, 2] >= image_w - self.edge_margin)
                | (boxes_tensor[:, 3] >= image_h - self.edge_margin)
            )
            if torch.any(touches_edge):
                return self._unchanged(sample)

        for _ in range(self.trials):
            scale = torch.empty(1).uniform_(self.min_scale, self.max_scale).item()
            crop_h = min(image_h, max(1, round(image_h * scale)))
            crop_w = min(image_w, max(1, round(image_w * scale)))

            max_top = image_h - crop_h
            max_left = image_w - crop_w
            top = int(torch.randint(max_top + 1, (1,)).item()) if max_top else 0
            left = int(torch.randint(max_left + 1, (1,)).item()) if max_left else 0

            clipped = boxes_tensor.clone()
            clipped[:, 0::2] = clipped[:, 0::2].clamp(min=left, max=left + crop_w) - left
            clipped[:, 1::2] = clipped[:, 1::2].clamp(min=top, max=top + crop_h) - top
            visible_ratio = self._box_area(clipped) / original_area

            if torch.all(visible_ratio >= self.min_visible_ratio):
                cropped_image = F.crop(image, top, left, crop_h, crop_w)
                cropped_target = target.copy()
                cropped_target["boxes"] = F.crop(boxes, top, left, crop_h, crop_w)
                cropped_target["area"] = self._box_area(
                    torch.as_tensor(cropped_target["boxes"], dtype=torch.float32)
                )
                cropped_target["size"] = torch.tensor([crop_h, crop_w])

                if "masks" in cropped_target:
                    cropped_target["masks"] = F.crop(
                        cropped_target["masks"], top, left, crop_h, crop_w
                    )

                return cropped_image, cropped_target, *extra

        return self._unchanged(sample)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt="", normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(
                inpt, key="boxes", box_format=self.fmt.upper(), spatial_size=spatial_size
            )

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype="float32", scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == "float32":
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.0

        inpt = Image(inpt)

        return inpt
