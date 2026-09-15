"""裁剪边界响应驱动的视域恢复。只处理原图像素坐标，不依赖模型、影像或真值。"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import asdict, dataclass, field, replace
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

Box = Tuple[float, float, float, float]
SIDES = ("left", "top", "right", "bottom")


@dataclass(frozen=True)
class Window:
    x0: int
    y0: int
    size: int

    @property
    def key(self) -> Tuple[int, int, int]:
        return self.x0, self.y0, self.size

    @property
    def window_id(self) -> str:
        return f"x{self.x0}_y{self.y0}_s{self.size}"

    @property
    def box(self) -> Box:
        return self.x0, self.y0, self.x0 + self.size, self.y0 + self.size


@dataclass(frozen=True)
class Observation:
    box: Box
    score: float
    window: Window
    class_id: int = 1
    stage: str = "base"
    view: str = "original"


@dataclass
class Prediction:
    box: Box
    score: float
    class_id: int = 1
    target_id: str = ""
    status: str = "unverified"
    support_count: int = 1
    refine_count: int = 0
    anchor_window: str = ""
    trusted_sides: Tuple[bool, bool, bool, bool] = (False,) * 4


@dataclass(frozen=True)
class StrategyConfig:
    # 所有默认阈值仅为首版实验起点；需在独立验证区域选择。
    base_size: int = 512
    stride: int = 256
    grid_offset: Tuple[int, int] = (0, 0)
    model_input_size: int = 512
    policy: str = "response"  # response/static/fixed/random/overlap
    scales: Tuple[int, ...] = (512, 768, 1024)
    shift_pixels: int = 64
    edge_margin_ratio: float = 0.0625
    stable_tolerance: float = 0.025  # 逐边误差除以目标对应宽或高
    response_min: float = 0.6
    response_max: float = 1.4
    candidate_confidence: float = 0.05
    refine_min_confidence: float = 0.15
    stable_confidence: float = 0.5
    match_iou: float = 0.35
    shared_iou: float = 0.65
    projection_iou: float = 0.85
    within_window_nms_iou: float = 0.9
    ambiguity_gap: float = 0.08
    max_area_ratio: float = 4.0
    nms_iou: float = 0.5
    max_refines_per_target: int = 3
    max_refine_windows: int = 1000
    min_object_pixels: float = 24.0
    use_side_fusion: bool = True
    random_seed: int = 3407

    def validate(self) -> None:
        integers = (self.base_size, self.stride, self.model_input_size, self.shift_pixels,
                    self.max_refines_per_target, self.max_refine_windows, self.random_seed,
                    *self.scales, *self.grid_offset)
        if any(not isinstance(v, int) or isinstance(v, bool) for v in integers):
            raise ValueError("窗口、位移、预算、随机种子和网格偏移必须是整数。")
        if self.base_size <= 0 or not 0 < self.stride <= self.base_size:
            raise ValueError("基础窗口和步长必须满足 0 < stride <= base_size。")
        if len(self.grid_offset) != 2 or any(not 0 <= v < self.stride for v in self.grid_offset):
            raise ValueError("grid_offset 两个分量必须位于 [0, stride)。")
        if self.model_input_size <= 0 or self.model_input_size % 32:
            raise ValueError("模型输入尺寸必须是正的32倍数。")
        if self.policy not in {"response", "static", "fixed", "random", "overlap"}:
            raise ValueError("未知推理策略。")
        if not self.scales or tuple(sorted(set(self.scales))) != self.scales or self.scales[0] != self.base_size:
            raise ValueError("scales 必须严格递增且以 base_size 开始。")
        if self.shift_pixels <= 0 or not 0 < self.edge_margin_ratio < 0.5:
            raise ValueError("位移应为正值；边距比例必须位于 (0, 0.5)。")
        probabilities = (self.candidate_confidence, self.refine_min_confidence,
                         self.stable_confidence, self.match_iou, self.shared_iou, self.nms_iou,
                         self.projection_iou, self.within_window_nms_iou)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in probabilities):
            raise ValueError("分数和IoU阈值必须位于 [0, 1]。")
        if any(v <= 0 for v in (self.match_iou, self.shared_iou, self.projection_iou,
                                self.nms_iou, self.within_window_nms_iou)):
            raise ValueError("IoU阈值必须大于0。")
        if not self.candidate_confidence <= self.refine_min_confidence <= self.stable_confidence:
            raise ValueError("阈值需满足 candidate <= refine_min <= stable_confidence。")
        if self.max_refines_per_target < 0 or self.max_refine_windows < 0:
            raise ValueError("复检预算不得为负数。")
        geometry = (self.max_area_ratio, self.min_object_pixels, self.stable_tolerance,
                    self.response_min, self.response_max, self.ambiguity_gap, self.edge_margin_ratio)
        if any(not math.isfinite(v) for v in geometry):
            raise ValueError("几何阈值必须是有限数值。")
        if self.max_area_ratio <= 1 or self.min_object_pixels <= 0 or self.stable_tolerance <= 0:
            raise ValueError("面积比上限应大于1，最小像素和稳定性容差应为正值。")
        if not 0 <= self.response_min < self.response_max or not 0 <= self.ambiguity_gap < 1:
            raise ValueError("边界响应区间或匹配歧义阈值不合法。")


@dataclass
class StrategyResult:
    predictions: List[Prediction]
    observations: List[Observation]
    events: List[dict]
    targets: List[dict]
    stats: dict


# Public runner API (implemented below):
# run_strategy(base_observations, base_windows, infer_window, width, height,
#              config=StrategyConfig(), progress=None) -> StrategyResult
# infer_window(Window) -> Sequence[Observation]; empty means no detection.
# base_windows includes ALL attempted base windows, including empty/nodata ones.
# The runner caches them so rechecks never spend a forward on an identical crop.


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _intersection(a: Box, b: Box) -> Box:
    return max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])


def box_iou(a: Box, b: Box) -> float:
    intersection = _area(_intersection(a, b))
    return intersection / max(_area(a) + _area(b) - intersection, 1e-12)


def grid_windows(width: int, height: int, size: int, stride: int,
                 offset: Tuple[int, int] = (0, 0)) -> List[Window]:
    """改变内部网格相位，额外保留0和末端窗口以保证整幅覆盖。"""
    if min(width, height, size, stride) <= 0 or stride > size:
        raise ValueError("影像、窗口、步长应为正且 stride <= size。")
    if len(offset) != 2 or any(not isinstance(v, int) or not 0 <= v < stride for v in offset):
        raise ValueError("网格偏移需为 [0,stride) 内的整数。")

    def starts(length: int, phase: int) -> List[int]:
        last = max(0, length - size)
        return sorted({0, last, *range(phase, last + 1, stride)})

    return [Window(x, y, size) for y in starts(height, offset[1])
            for x in starts(width, offset[0])]


def _margins(obs: Observation) -> Tuple[float, ...]:
    b, w = obs.box, obs.window.box
    return b[0] - w[0], b[1] - w[1], w[2] - b[2], w[3] - b[3]


def _safe_sides(obs: Observation, cfg: StrategyConfig) -> Tuple[bool, ...]:
    return tuple(v >= cfg.edge_margin_ratio * obs.window.size for v in _margins(obs))


def _external_sides(obs: Observation, width: int, height: int,
                    cfg: StrategyConfig) -> Tuple[bool, ...]:
    # 真实影像边界附近没有可读外部上下文；padding不算新增观测。
    b, w = obs.box, obs.window.box
    margin = cfg.edge_margin_ratio * obs.window.size
    return (w[0] <= 0 and b[0] < margin,
            w[1] <= 0 and b[1] < margin,
            w[2] >= width and width - b[2] < margin,
            w[3] >= height and height - b[3] < margin)


def _internal_risk(obs: Observation, width: int, height: int,
                   cfg: StrategyConfig) -> Tuple[bool, ...]:
    return tuple(not safe and not external for safe, external in
                 zip(_safe_sides(obs, cfg), _external_sides(obs, width, height, cfg)))


def _projection_score(partial: Observation, larger: Observation, width: int,
                      height: int, cfg: StrategyConfig) -> float:
    """较大观测投回片段来源窗口，应解释片段且确实越过其人工截断边。"""
    if partial.class_id != larger.class_id or _area(larger.box) <= _area(partial.box):
        return 0.0
    risk = _internal_risk(partial, width, height, cfg)
    w, b = partial.window.box, larger.box
    crosses = (b[0] < w[0], b[1] < w[1], b[2] > w[2], b[3] > w[3])
    if not any(r and c for r, c in zip(risk, crosses)):
        return 0.0
    projection = _intersection(b, w)
    score = box_iou(partial.box, projection)
    return score if score >= cfg.projection_iou else 0.0


def association_score(first: Observation, second: Observation, width: int,
                      height: int, config: StrategyConfig) -> float:
    """固定锚点关联，或在共同可见区域验证裁剪片段；不做连通分量串联合并。"""
    if first.class_id != second.class_id:
        return 0.0
    a, b = first.box, second.box
    projection = max(_projection_score(first, second, width, height, config),
                     _projection_score(second, first, width, height, config))
    if projection:
        return max(box_iou(a, b), 0.95 * projection)
    ratio = max(_area(a), _area(b)) / max(min(_area(a), _area(b)), 1e-12)
    if ratio > config.max_area_ratio:
        return 0.0
    direct = box_iou(a, b)
    if direct >= config.match_iou:
        return direct
    if not (any(_internal_risk(first, width, height, config)) or
            any(_internal_risk(second, width, height, config))):
        return 0.0
    common = _intersection(first.window.box, second.window.box)
    shared_a, shared_b = _intersection(a, common), _intersection(b, common)
    if min(_area(shared_a), _area(shared_b)) < 0.2 * min(_area(a), _area(b)):
        return 0.0
    shared = box_iou(shared_a, shared_b)
    if shared < config.shared_iou:
        return 0.0
    # 共享区之外仍需有实际重合，避免把相邻厂区仅凭一条裁剪线拼起来。
    ios = _area(_intersection(a, b)) / max(min(_area(a), _area(b)), 1e-12)
    if ios < config.shared_iou:
        return 0.0
    return min(0.8, shared * ios)


def boundary_response(first: Observation, second: Observation, width: int,
                      height: int, config: StrategyConfig = StrategyConfig()) -> dict:
    """四边顺序L/T/R/B；只对同尺度、不同裁剪、原始视图计算响应。"""
    result = {"comparable": False, "ratios": [None] * 4,
              "following": [False] * 4, "stable_sides": [False] * 4}
    if (first.window.size != second.window.size or first.window.key == second.window.key
            or first.view != "original" or second.view != "original"
            or not association_score(first, second, width, height, config)):
        return result
    result["comparable"] = True
    risks_a = _internal_risk(first, width, height, config)
    risks_b = _internal_risk(second, width, height, config)
    safe_a, safe_b = _safe_sides(first, config), _safe_sides(second, config)
    external_a = _external_sides(first, width, height, config)
    external_b = _external_sides(second, width, height, config)
    extents = [(first.box[2] - first.box[0] + second.box[2] - second.box[0]) / 2,
               (first.box[3] - first.box[1] + second.box[3] - second.box[1]) / 2]
    for side in range(4):
        delta = second.window.box[side] - first.window.box[side]
        db = second.box[side] - first.box[side]
        if abs(delta) >= 1:
            response = db / delta
            result["ratios"][side] = response
            result["following"][side] = bool(
                risks_a[side] and risks_b[side] and
                config.response_min <= response <= config.response_max)
        result["stable_sides"][side] = bool(
            safe_a[side] and safe_b[side] and not external_a[side] and not external_b[side]
            and abs(db) / max(extents[side % 2], 1.0) <= config.stable_tolerance)
    return result


class _SpatialIndex:
    def __init__(self, cell_size: int):
        self.cell_size = cell_size
        self.cells: Dict[Tuple[int, int], set] = defaultdict(set)

    def _keys(self, box: Box) -> Iterable[Tuple[int, int]]:
        if not _area(box):
            return
        for y in range(math.floor(box[1] / self.cell_size), math.floor((box[3] - 1e-6) / self.cell_size) + 1):
            for x in range(math.floor(box[0] / self.cell_size), math.floor((box[2] - 1e-6) / self.cell_size) + 1):
                yield x, y

    def add(self, index: int, box: Box) -> None:
        for key in self._keys(box):
            self.cells[key].add(index)

    def query(self, box: Box) -> List[int]:
        found: set = set()
        for key in self._keys(box):
            found.update(self.cells.get(key, ()))
        return sorted(found)


@dataclass
class _Target:
    target_id: str
    anchor: Observation
    observations: List[Observation] = field(default_factory=list)
    requested: set = field(default_factory=set)
    requests: int = 0
    forward_count: int = 0
    status: str = "unverified"
    reason: str = ""
    ambiguous: bool = False


def _quality(obs: Observation, width: int, height: int, cfg: StrategyConfig) -> tuple:
    safe = _safe_sides(obs, cfg)
    external = _external_sides(obs, width, height, cfg)
    # 优先安全边数，随后用原模型置信度；不使用框面积作为完整性证据。
    return sum(s and not e for s, e in zip(safe, external)), obs.score


def _best(target: _Target, width: int, height: int, cfg: StrategyConfig) -> Observation:
    return max(target.observations, key=lambda o: _quality(o, width, height, cfg))


def _analyse(target: _Target, width: int, height: int, cfg: StrategyConfig) -> dict:
    trusted = {o: [False] * 4 for o in target.observations}
    pairs = []
    current = target.observations[-1] if target.requests else _best(target, width, height, cfg)
    following = [False] * 4
    for i, first in enumerate(target.observations):
        for second in target.observations[i + 1:]:
            response = boundary_response(first, second, width, height, cfg)
            if not response["comparable"]:
                continue
            pairs.append({"first": first.window.window_id, "second": second.window.window_id, **response})
            for side, stable in enumerate(response["stable_sides"]):
                trusted[first][side] |= stable
                trusted[second][side] |= stable
            if current in (first, second):
                following = [a or b for a, b in zip(following, response["following"])]
    complete = [o for o in target.observations if all(trusted[o])]
    anchor = (max(complete, key=lambda o: o.score) if complete
              else _best(target, width, height, cfg))
    return {"trusted": trusted, "pairs": pairs, "following": following,
            "current": current, "complete": complete, "anchor": anchor}


def _build_targets(observations: Sequence[Observation], width: int, height: int,
                   cfg: StrategyConfig) -> Tuple[List[_Target], _SpatialIndex]:
    by_window: Dict[Tuple[int, int, int], List[Observation]] = defaultdict(list)
    for obs in observations:
        by_window[obs.window.key].append(obs)
    targets: List[_Target] = []
    index = _SpatialIndex(cfg.base_size)
    def window_priority(key: tuple) -> tuple:
        best = max(by_window[key], key=lambda o: _quality(o, width, height, cfg))
        return (-_quality(best, width, height, cfg)[0], -best.score, key)

    for key in sorted(by_window, key=window_priority):
        group = sorted(by_window[key], key=lambda o: (-o.score, o.box))
        proposals: Dict[int, List[Tuple[float, int]]] = defaultdict(list)
        uncertain = set()
        for oi, obs in enumerate(group):
            candidates = []
            for ti in index.query(obs.box):
                t = targets[ti]
                if any(o.window.key == key for o in t.observations):
                    continue
                score = association_score(t.anchor, obs, width, height, cfg)
                if score:
                    candidates.append((score, ti))
            candidates.sort(reverse=True)
            if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < cfg.ambiguity_gap:
                uncertain.add(oi)
                for _, ti in candidates[:2]:
                    targets[ti].ambiguous = True
            elif candidates:
                proposals[candidates[0][1]].append((candidates[0][0], oi))
        assignments = {}
        for ti, offers in proposals.items():
            offers.sort(reverse=True)
            if len(offers) > 1 and offers[0][0] - offers[1][0] < cfg.ambiguity_gap:
                targets[ti].ambiguous = True
                uncertain.update(oi for _, oi in offers)
            else:
                assignments[offers[0][1]] = ti
        for oi, obs in enumerate(group):
            if oi in assignments:
                targets[assignments[oi]].observations.append(obs)
            else:
                target = _Target(f"T{len(targets) + 1:06d}", obs, [obs], ambiguous=oi in uncertain)
                index.add(len(targets), obs.box)
                targets.append(target)
    return targets, index


def _bounded_window(cx: float, cy: float, size: int, width: int, height: int) -> Window:
    return Window(min(max(0, int(round(cx - size / 2))), max(0, width - size)),
                  min(max(0, int(round(cy - size / 2))), max(0, height - size)), size)


def _allowed_window(window: Window, current: Observation, target: _Target,
                    width: int, height: int, cfg: StrategyConfig) -> bool:
    known = {o.window.key for o in target.observations} | target.requested
    if window.key in known:
        return False
    # 扩域不能把目标短边压缩到低于设定的模型输入像素。
    if window.size > current.window.size:
        short_side = min(current.box[2] - current.box[0], current.box[3] - current.box[1])
        if short_side * cfg.model_input_size / window.size < cfg.min_object_pixels:
            return False
    valid = _intersection(window.box, (0, 0, width, height))
    if _area(_intersection(current.box, valid)) / _area(current.box) < 0.95:
        return False
    return True


def _next_window(target: _Target, analysis: dict, width: int, height: int,
                 cfg: StrategyConfig, rng: random.Random) -> Tuple[Optional[Window], str]:
    current = analysis["current"]
    b, w = current.box, current.window
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    step = max(1, round(cfg.shift_pixels * w.size / cfg.base_size))
    risk = _internal_risk(current, width, height, cfg)
    external = _external_sides(current, width, height, cfg)
    proposed: List[Tuple[Window, str]] = []

    def shift(dx: int, dy: int, reason: str) -> None:
        proposed.append((_bounded_window(w.x0 + w.size / 2 + dx,
                                         w.y0 + w.size / 2 + dy,
                                         w.size, width, height), reason))

    def expand(reason: str) -> None:
        for size in cfg.scales:
            if size > w.size:
                # 缺失方向给额外上下文，且仍完整包含当前可见框。
                dx = (int(risk[2]) - int(risk[0])) * (size - w.size) / 4
                dy = (int(risk[3]) - int(risk[1])) * (size - w.size) / 4
                proposed.append((_bounded_window(cx + dx, cy + dy, size, width, height), reason))

    if cfg.policy == "fixed":
        size = cfg.scales[min(1, len(cfg.scales) - 1)]
        seed = target.anchor.box
        scx, scy = (seed[0] + seed[2]) / 2, (seed[1] + seed[3]) / 2
        for dx, dy in ((0, 0), (cfg.shift_pixels, 0), (0, cfg.shift_pixels), (-cfg.shift_pixels, 0)):
            proposed.append((_bounded_window(scx + dx, scy + dy, size, width, height), "fixed_context"))
    elif cfg.policy == "random":
        for _ in range(24):
            size = rng.choice(cfg.scales)
            proposed.append((_bounded_window(cx + rng.uniform(-step, step),
                                              cy + rng.uniform(-step, step), size, width, height), "random_view"))
    elif cfg.policy == "static":
        if any(risk):
            proposed.append((_bounded_window(cx, cy, w.size, width, height), "static_recenter"))
            expand("static_expand")
        else:
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step)):
                shift(dx, dy, "verify_stability")
    else:
        follows = any(analysis["following"])
        opposite = (risk[0] and risk[2]) or (risk[1] and risk[3])
        if follows or opposite:
            expand("boundary_following_expand" if follows else "opposite_sides_expand")
        elif any(risk):
            shift((int(risk[2]) - int(risk[0])) * step,
                  (int(risk[3]) - int(risk[1])) * step, "directional_probe")
            expand("no_available_shift_expand")
        else:
            # 完整性确认需第二个真实裁剪；TTA不算独立的空间支持。
            for dx, dy in ((step, 0), (-step, 0), (0, step), (0, -step), (step, step)):
                shift(dx, dy, "verify_stability")
            if not any(external):
                expand("verification_context_expand")
    for window, reason in proposed:
        if _allowed_window(window, current, target, width, height, cfg):
            if reason == "verify_stability":
                hypothetical = replace(current, window=window)
                if not all(_safe_sides(hypothetical, cfg)):
                    continue
            return window, reason
    return None, "image_boundary" if any(external) else "scale_limit"


def _match_refinement(target: _Target, candidates: Sequence[Observation], targets: Sequence[_Target],
                      index: _SpatialIndex, width: int, height: int,
                      cfg: StrategyConfig) -> Tuple[Optional[Observation], str]:
    scored = []
    for obs in candidates:
        # 始终与初始锚点建立关联，防止连续扩框发生链式漂移。
        score = association_score(target.anchor, obs, width, height, cfg)
        if score:
            scored.append((score, obs.score, obs))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    if not scored:
        return None, "not_found"
    if len(scored) > 1 and scored[0][0] - scored[1][0] < cfg.ambiguity_gap:
        return None, "ambiguous"
    score, _, obs = scored[0]
    for ti in index.query(obs.box):
        other = targets[ti]
        if other is target:
            continue
        competitor = association_score(other.anchor, obs, width, height, cfg)
        if competitor and competitor >= score - cfg.ambiguity_gap:
            if (_projection_score(other.anchor, obs, width, height, cfg) and
                    not _coobserved_separately(target, other, cfg)):
                # 同厂另一截断片段不应反过来阻止完整观测；最终仍需唯一可靠投影才能吸收。
                continue
            return None, "ambiguous"
    return obs, "matched"


def _coobserved_separately(first: _Target, second: _Target, cfg: StrategyConfig) -> bool:
    return any(a.window.key == b.window.key and box_iou(a.box, b.box) < cfg.within_window_nms_iou
               for a in first.observations for b in second.observations)


def _target_prediction(target: _Target, width: int, height: int, cfg: StrategyConfig) -> Prediction:
    analysis = _analyse(target, width, height, cfg)
    anchor = analysis["anchor"]
    trusted = analysis["trusted"]
    box = list(anchor.box)
    if cfg.use_side_fusion:
        # 只在已有完整可靠观测时微调边界，绝不把几个局部框的min/max拼成新厂区。
        if analysis["complete"]:
            for side in range(4):
                members = [o for o in analysis["complete"] if trusted[o][side]]
                denominator = sum(o.score for o in members)
                if denominator > 0:
                    box[side] = sum(o.box[side] * o.score for o in members) / denominator
    else:
        # 消融：同一已可靠关联目标内的普通置信度加权整框平均。
        denominator = sum(o.score for o in target.observations)
        if denominator > 0:
            box = [sum(o.box[side] * o.score for o in target.observations) / denominator
                   for side in range(4)]
    return Prediction(tuple(box), anchor.score, anchor.class_id, target.target_id,
                      target.status, len({o.window.key for o in target.observations}),
                      target.requests, anchor.window.window_id, tuple(trusted[anchor]))


def _nms_predictions(predictions: Sequence[Prediction], cfg: StrategyConfig) -> List[Prediction]:
    # 常规全局NMS仅解决重复；低IoU片段已由可靠目标关联替换，未使用宽松IoS盲并。
    ordered = sorted(predictions, key=lambda p: (-p.score, p.target_id))
    kept: List[Prediction] = []
    index = _SpatialIndex(cfg.base_size)
    for pred in ordered:
        if any(pred.class_id == kept[i].class_id and box_iou(pred.box, kept[i].box) >= cfg.nms_iou
               for i in index.query(pred.box)):
            continue
        index.add(len(kept), pred.box)
        kept.append(pred)
    return kept


def _deduplicate_observations(observations: Sequence[Observation], cfg: StrategyConfig) -> List[Observation]:
    """统一所有策略的同窗高重合query去重，避免低分重复query制造关联歧义。"""
    groups: Dict[tuple, List[Observation]] = defaultdict(list)
    for obs in observations:
        groups[(obs.window.key, obs.view, obs.class_id)].append(obs)
    clean = []
    for group in groups.values():
        kept: List[Observation] = []
        for obs in sorted(group, key=lambda o: (-o.score, o.box)):
            if not any(box_iou(obs.box, old.box) >= cfg.within_window_nms_iou for old in kept):
                kept.append(obs)
        clean.extend(kept)
    return clean


def _absorb_explained_fragments(predictions: Sequence[Prediction], targets: Sequence[_Target],
                               width: int, height: int, cfg: StrategyConfig) -> Tuple[List[Prediction], dict]:
    lookup = {t.target_id: t for t in targets}
    reliable = [p for p in predictions if all(p.trusted_sides) and p.target_id in lookup]
    # 同一完整目标可能由两条片段恢复路径得到；先去掉近乎相同的完整解释，
    # 否则一片段同时匹配两个相同完整框会被误认为关联歧义。
    reliable = _nms_predictions(reliable, replace(cfg, nms_iou=cfg.within_window_nms_iou))
    retained = []
    absorbed = {}
    for pred in predictions:
        target = lookup.get(pred.target_id)
        if target is None or all(pred.trusted_sides):
            retained.append(pred)
            continue
        explaining = []
        for complete in reliable:
            other = lookup[complete.target_id]
            if other is target or _coobserved_separately(target, other, cfg):
                continue
            full = replace(_best(other, width, height, cfg), box=complete.box)
            if _projection_score(target.anchor, full, width, height, cfg):
                explaining.append(complete.target_id)
        if len(explaining) == 1:
            absorbed[pred.target_id] = explaining[0]
        else:
            retained.append(pred)
    return retained, absorbed


def _validate_observations(observations: Sequence[Observation], width: int, height: int,
                           cfg: StrategyConfig) -> List[Observation]:
    valid = []
    for obs in observations:
        if (len(obs.box) != 4 or any(not math.isfinite(v) for v in obs.box)
                or not math.isfinite(obs.score) or not 0 <= obs.score <= 1
                or obs.box[2] <= obs.box[0] or obs.box[3] <= obs.box[1]
                or obs.window.size <= 0 or obs.window.x0 < 0 or obs.window.y0 < 0):
            raise ValueError("观测包含非法窗口、坐标或置信度。")
        visible = _intersection(obs.window.box, (0, 0, width, height))
        if any(abs(a - b) > 1e-4 for a, b in zip(_intersection(obs.box, visible), obs.box)):
            raise ValueError("观测必须已裁至其窗口与影像的共同可见范围。")
        if obs.score >= cfg.candidate_confidence:
            valid.append(replace(obs, box=tuple(float(v) for v in obs.box)))
    return valid


def run_strategy(base_observations: Sequence[Observation], base_windows: Sequence[Window],
                 infer_window: Callable[[Window], Sequence[Observation]], width: int, height: int,
                 config: StrategyConfig = StrategyConfig(),
                 progress: Optional[Callable[[str], None]] = None) -> StrategyResult:
    """无真值的有预算推理。缓存以真实裁剪(x,y,size)标识，不以查询索引关联。"""
    config.validate()
    if (not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0):
        raise ValueError("影像尺寸必须为正。")
    for window in base_windows:
        if (any(not isinstance(v, int) or isinstance(v, bool) for v in window.key) or
                window.size <= 0 or window.x0 < 0 or window.y0 < 0 or
                window.x0 >= width or window.y0 >= height):
            raise ValueError("基础窗口必须具有正尺寸且起点位于影像内。")
    raw_observations = _validate_observations(base_observations, width, height, config)
    observations = _deduplicate_observations(raw_observations, config)
    cache: Dict[Tuple[int, int, int], List[Observation]] = {w.key: [] for w in base_windows}
    for obs in observations:
        cache.setdefault(obs.window.key, []).append(obs)
    stats = {"base_windows": len(cache), "base_observations": len(observations),
             "refine_windows": 0, "cache_hits": 0, "requested_views": 0,
             "policy": config.policy, "base_duplicate_queries_removed": len(raw_observations) - len(observations)}
    events: List[dict] = []
    if config.policy == "overlap":
        predictions = [Prediction(o.box, o.score, o.class_id, f"B{i:07d}",
                                  anchor_window=o.window.window_id) for i, o in enumerate(observations)]
        predictions = _nms_predictions(predictions, config)
        return StrategyResult(predictions, observations, events, [],
                              {**stats, "output_candidates": len(predictions)})
    targets, index = _build_targets(observations, width, height, config)
    rng = random.Random(config.random_seed)
    # 只按可用预测证据分配预算；真实目标数量/GT永不进入排序。
    def priority(t: _Target) -> tuple:
        o = _best(t, width, height, config)
        risk = sum(_internal_risk(o, width, height, config))
        uncertain = int(o.score < config.stable_confidence)
        return (-(risk + uncertain), -o.score, t.target_id)

    consumed = set(observations)
    unmatched_refinements = []
    for ti, target in enumerate(sorted(targets, key=priority)):
        best = _best(target, width, height, config)
        if target.ambiguous:
            target.status, target.reason = "ambiguous", "base_association_ambiguous"
        elif best.score < config.refine_min_confidence:
            target.reason = "below_refine_threshold"
        else:
            while True:
                analysis = _analyse(target, width, height, config)
                if analysis["complete"]:
                    target.status = "recovered" if target.requests else "stable"
                    target.reason = "safe_and_stable_distinct_crops"
                    break
                current = analysis["current"]
                if (any(_external_sides(current, width, height, config)) and
                        not any(_internal_risk(current, width, height, config))):
                    target.status, target.reason = "image_boundary", "outside_raster_context_unavailable"
                    break
                if target.requests >= config.max_refines_per_target:
                    target.status, target.reason = "budget_exhausted", "per_target_budget"
                    break
                window, action = _next_window(target, analysis, width, height, config, rng)
                if window is None:
                    target.status, target.reason = action, action
                    break
                cached = window.key in cache
                if not cached and stats["refine_windows"] >= config.max_refine_windows:
                    target.status, target.reason = "budget_exhausted", "global_budget"
                    break
                target.requests += 1
                target.requested.add(window.key)
                stats["requested_views"] += 1
                if cached:
                    stats["cache_hits"] += 1
                else:
                    new = _validate_observations(list(infer_window(window)), width, height, config)
                    new = _deduplicate_observations(new, config)
                    if any(o.window.key != window.key for o in new):
                        raise ValueError("infer_window 返回的观测不属于所请求窗口。")
                    cache[window.key] = new
                    observations.extend(new)
                    unmatched_refinements.extend(new)
                    stats["refine_windows"] += 1
                    target.forward_count += 1
                matched, outcome = _match_refinement(target, cache[window.key], targets,
                                                     index, width, height, config)
                event = {"target_id": target.target_id, "action": action,
                         "window": asdict(window), "cached": cached, "outcome": outcome,
                         "per_target_requests": target.requests,
                         "global_refine_windows": stats["refine_windows"],
                         "trigger_following": dict(zip(SIDES, analysis["following"])),
                         "response": None}
                if matched is None:
                    target.status, target.reason = outcome, "refinement_" + outcome
                    events.append(event)
                    break  # 未匹配/有歧义不靠扩大视野持续猜测；保留初始结果。
                consumed.add(matched)
                event["response"] = boundary_response(current, matched, width, height, config)
                event["matched_box"] = list(matched.box)
                event["matched_score"] = matched.score
                target.observations.append(matched)
                events.append(event)
                if progress:
                    progress(f"[复检] {target.target_id} {action} | {outcome} | "
                             f"新增窗口 {stats['refine_windows']}/{config.max_refine_windows}")
        if progress and (ti + 1) % 100 == 0:
            progress(f"[目标] 已处理 {ti + 1}/{len(targets)}")
    predictions = [_target_prediction(t, width, height, config) for t in targets]
    # 扩域视图中意外检出的其他目标仍可进入结果，但不沿它们继续递归搜索。
    for i, obs in enumerate(unmatched_refinements):
        if obs not in consumed:
            if any(association_score(targets[i].anchor, obs, width, height, config)
                   for i in index.query(obs.box)):
                # 与既有目标相关却未唯一匹配的候选不作为“新厂区”绕过歧义保护。
                continue
            predictions.append(Prediction(obs.box, obs.score, obs.class_id, f"R{i:07d}",
                                          "unverified", anchor_window=obs.window.window_id))
    predictions, absorbed = _absorb_explained_fragments(predictions, targets, width, height, config)
    predictions = _nms_predictions(predictions, config)
    exported_target_ids = {p.target_id for p in predictions}
    reports = []
    for target in targets:
        analysis = _analyse(target, width, height, config)
        reports.append({"target_id": target.target_id, "status": target.status,
                        "stop_reason": target.reason, "requested_views": target.requests,
                        "new_windows": target.forward_count, "anchor_box": list(target.anchor.box),
                        "selected_box": list(analysis["anchor"].box),
                        "support_windows": [o.window.window_id for o in target.observations],
                        "responses": analysis["pairs"],
                        "absorbed_by": absorbed.get(target.target_id),
                        "retained_after_nms": target.target_id in exported_target_ids})
    counts: Dict[str, int] = defaultdict(int)
    for prediction in predictions:
        counts[prediction.status] += 1
    stats.update({"targets": len(targets), "output_candidates": len(predictions),
                  "output_status_counts": dict(counts),
                  "budget_note": "global=新裁剪回调次数(含无效窗口); per_target=请求视图数(含缓存)",
                  "config": asdict(config)})
    return StrategyResult(predictions, observations, events, reports, stats)
