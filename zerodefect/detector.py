"""Golden-sample anomaly detector.

Zero training: a frozen ImageNet backbone (ResNet-18, ONNX) turns the golden sample(s)
into a memory bank of local patch features (PatchCore-style). A part under inspection is
scored by how far each of its patches is from the nearest "good" patch. No defect images
and no gradient updates are ever needed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

try:
    import onnx
    import onnxruntime as ort
except ImportError:  # pragma: no cover - surfaced at runtime
    onnx = None
    ort = None

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Node names in the ONNX model zoo's resnet18-v1-7 graph (end of layer2 / layer3).
LAYER2 = "resnetv15_stage2_activation1"
LAYER3 = "resnetv15_stage3_activation1"


class FeatureExtractor:
    """Runs the backbone up to layer3 and returns a dense patch-feature grid."""

    def __init__(self, model_path: str | Path, input_size: int = 256):
        if onnx is None or ort is None:
            raise RuntimeError("onnx and onnxruntime are required: pip install onnx onnxruntime")
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Backbone not found at {model_path}. Run: python scripts/download_model.py"
            )
        self.input_size = int(input_size)
        model = onnx.load(str(model_path))
        graph = model.graph

        # Keep only the nodes needed to compute layer2 + layer3 (drop the classifier head).
        keep, needed = [], {LAYER2, LAYER3}
        for node in reversed(graph.node):
            if any(o in needed for o in node.output):
                keep.append(node)
                needed.update(i for i in node.input if i)
        keep.reverse()
        del graph.node[:]
        graph.node.extend(keep)
        del graph.output[:]
        graph.output.extend(
            [
                onnx.helper.make_tensor_value_info(LAYER2, onnx.TensorProto.FLOAT, None),
                onnx.helper.make_tensor_value_info(LAYER3, onnx.TensorProto.FLOAT, None),
            ]
        )
        # Make the spatial input size dynamic.
        dims = graph.input[0].type.tensor_type.shape.dim
        dims[0].dim_param = "N"
        dims[2].dim_param = "H"
        dims[3].dim_param = "W"
        del graph.value_info[:]

        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            model.SerializeToString(), opts, providers=["CPUExecutionProvider"]
        )
        self.input_name = graph.input[0].name

    def preprocess(self, images: list[np.ndarray]) -> np.ndarray:
        batch = []
        for img in images:
            x = cv2.resize(img, (self.input_size, self.input_size), interpolation=cv2.INTER_AREA)
            x = x.astype(np.float32) / 255.0
            x = (x - IMAGENET_MEAN) / IMAGENET_STD
            batch.append(x.transpose(2, 0, 1))
        return np.ascontiguousarray(np.stack(batch), dtype=np.float32)

    def __call__(self, images: list[np.ndarray], batch_size: int = 8) -> np.ndarray:
        """images: list of RGB uint8 arrays. Returns (N, h, w, C) patch features."""
        out = []
        for i in range(0, len(images), batch_size):
            x = self.preprocess(images[i : i + batch_size])
            f2, f3 = self.session.run([LAYER2, LAYER3], {self.input_name: x})
            out.append(_combine(f2, f3))
        return np.concatenate(out, axis=0)


def _avg_pool3(x: np.ndarray) -> np.ndarray:
    """3x3 local neighbourhood aggregation (PatchCore), x: (N, C, H, W)."""
    n, c, h, w = x.shape
    flat = x.reshape(n * c, h, w)
    pooled = np.stack([cv2.blur(ch, (3, 3), borderType=cv2.BORDER_REFLECT) for ch in flat])
    return pooled.reshape(n, c, h, w)


def _combine(f2: np.ndarray, f3: np.ndarray) -> np.ndarray:
    f2 = _avg_pool3(f2)
    f3 = _avg_pool3(f3)
    n, _, h, w = f2.shape
    up = np.stack(
        [
            np.stack([cv2.resize(ch, (w, h), interpolation=cv2.INTER_LINEAR) for ch in sample])
            for sample in f3
        ]
    )
    feats = np.concatenate([f2, up], axis=1)  # (N, 384, h, w)
    return feats.transpose(0, 2, 3, 1).astype(np.float32)


# ----------------------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------------------

def _rotate(img: np.ndarray, angle: float, scale: float = 1.0, shift=(0.0, 0.0)) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    m[0, 2] += shift[0] * w
    m[1, 2] += shift[1] * h
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _brightness(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def _sensor(img: np.ndarray, rng: np.random.Generator, noise: float, blur: float, jpeg: int) -> np.ndarray:
    """Re-shoot the same part: sensor noise, focus blur and phone-camera JPEG compression."""
    x = img.astype(np.float32) + rng.normal(0, noise, img.shape)
    if blur > 0:
        x = cv2.GaussianBlur(x, (0, 0), blur)
    x = np.clip(x, 0, 255).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, jpeg])
    return cv2.imdecode(enc, cv2.IMREAD_UNCHANGED) if ok else x


def augment_golden(img: np.ndarray, rotation_invariant: bool) -> list[np.ndarray]:
    """Nuisance variations a phone camera will introduce (pose jitter, lighting, sensor)."""
    rng = np.random.default_rng(7)
    variants = [img]
    for noise, blur, jpeg in ((4, 0.0, 85), (6, 0.8, 75), (3, 1.2, 90)):
        variants.append(_sensor(img, rng, noise, blur, jpeg))
    for angle in (-4, 4):
        variants.append(_rotate(img, angle))
    for dx, dy in ((0.02, 0), (-0.02, 0), (0, 0.02), (0, -0.02)):
        variants.append(_rotate(img, 0, shift=(dx, dy)))
    variants.append(_rotate(img, 0, scale=1.03))
    variants.append(_rotate(img, 0, scale=0.97))
    for f in (0.9, 1.1):
        variants.append(_brightness(img, f))
    if rotation_invariant:
        for k in (1, 2, 3):
            base = np.ascontiguousarray(np.rot90(img, k))
            variants.extend([base, _rotate(base, 3), _rotate(base, -3)])
        for angle in (45, 135, 225, 315):
            variants.append(_rotate(img, angle))
    return variants


def calibration_views(img: np.ndarray) -> list[np.ndarray]:
    """Held-out good views (parameters not used in the bank) to set the pass/fail threshold."""
    rng = np.random.default_rng(11)
    return [
        _sensor(_rotate(img, 2, shift=(0.01, -0.01)), rng, 5, 0.5, 80),
        _sensor(_rotate(img, -2, shift=(-0.01, 0.01)), rng, 5, 1.0, 80),
        _sensor(_rotate(img, 1, scale=1.015), rng, 7, 0.0, 70),
        _sensor(_brightness(_rotate(img, -1), 0.95), rng, 5, 0.7, 80),
        _sensor(_brightness(img, 1.05), rng, 6, 0.3, 75),
        _sensor(_rotate(img, 3, scale=0.985), rng, 5, 0.6, 80),
    ]


def greedy_coreset(feats: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """k-center greedy subsampling (PatchCore) on a random projection for speed."""
    if len(feats) <= n:
        return feats
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((feats.shape[1], 128)).astype(np.float32) / math.sqrt(128)
    z = feats @ proj
    selected = [int(rng.integers(len(z)))]
    min_d = np.linalg.norm(z - z[selected[0]], axis=1)
    for _ in range(n - 1):
        idx = int(np.argmax(min_d))
        selected.append(idx)
        min_d = np.minimum(min_d, np.linalg.norm(z - z[idx], axis=1))
    return feats[np.array(selected)]


def nn_distance(queries: np.ndarray, bank: np.ndarray, bank_sq: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Euclidean distance from each query to its nearest bank vector."""
    out = np.empty(len(queries), dtype=np.float32)
    for i in range(0, len(queries), chunk):
        q = queries[i : i + chunk]
        d2 = (q * q).sum(1, keepdims=True) - 2.0 * q @ bank.T + bank_sq[None, :]
        out[i : i + chunk] = np.sqrt(np.maximum(d2.min(1), 0.0))
    return out


# ----------------------------------------------------------------------------------------
# The model
# ----------------------------------------------------------------------------------------

@dataclass
class Region:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in crop pixels
    peak: float  # peak normalised anomaly (1.0 == threshold)
    area_px: int

    def to_dict(self) -> dict:
        return {"box": list(self.box), "peak": round(float(self.peak), 3), "area_px": int(self.area_px)}


@dataclass
class ScoreResult:
    score: float  # normalised: >1.0 means FAIL
    raw_score: float
    anomaly_map: np.ndarray  # float32 (H, W), normalised to threshold
    regions: list[Region] = field(default_factory=list)

    @property
    def is_defect(self) -> bool:
        return self.score > 1.0


class GoldenSampleModel:
    """Learn "what good looks like" from one (or a few) golden samples."""

    def __init__(
        self,
        extractor: FeatureExtractor,
        size: int = 256,
        rotation_invariant: bool = False,
        max_bank: int = 30000,
        sensitivity: float = 1.0,
    ):
        self.extractor = extractor
        self.size = size
        self.rotation_invariant = rotation_invariant
        self.max_bank = max_bank
        self.sensitivity = sensitivity
        self.bank: np.ndarray | None = None
        self.bank_sq: np.ndarray | None = None
        self.threshold: float = 1.0
        self.auto_threshold: float = 1.0
        self.calibrated_with = 0
        self.golden: list[np.ndarray] = []

    # -- fitting ---------------------------------------------------------------------------
    def fit(self, goldens: list[np.ndarray]) -> "GoldenSampleModel":
        if not goldens:
            raise ValueError("Need at least one golden sample")
        self.golden = [self._prep(g) for g in goldens]
        views = []
        for g in self.golden:
            views.extend(augment_golden(g, self.rotation_invariant))
        feats = self.extractor(views)
        bank = feats.reshape(-1, feats.shape[-1])
        if len(bank) > self.max_bank:
            # Small greedy k-center coreset (covers rare patches) + random fill keeps fitting fast.
            rng = np.random.default_rng(0)
            keep = min(1500, self.max_bank // 4)
            pre = bank[rng.choice(len(bank), min(len(bank), 12000), replace=False)]
            core = greedy_coreset(pre, keep)
            rest = bank[rng.choice(len(bank), self.max_bank - keep, replace=False)]
            bank = np.concatenate([core, rest])
        self.bank = np.ascontiguousarray(bank, dtype=np.float32)
        self.bank_sq = (self.bank * self.bank).sum(1)

        # Threshold from held-out good views: the worst good patch score, plus a margin.
        cal = []
        for g in self.golden:
            cal.extend(calibration_views(g))
        cal_scores = [self._raw_map(f).max() for f in self.extractor(cal)]
        base = float(np.max(cal_scores))
        spread = float(np.std(cal_scores)) if len(cal_scores) > 1 else 0.0
        self.auto_threshold = (base + spread) * 0.9
        self.threshold = self.auto_threshold / max(self.sensitivity, 1e-3)
        self.calibrated_with = 0
        return self

    def calibrate(self, good_parts: list[np.ndarray], margin: float = 1.12) -> float:
        """Optional: show a few more OK parts; the threshold sits just above the worst of them."""
        if not good_parts:
            return self.threshold
        raws = [self._raw_map(f).max() for f in self.extractor([self._prep(p) for p in good_parts])]
        self.threshold = max(float(np.max(raws)) * margin, 0.8 * self.auto_threshold) / max(self.sensitivity, 1e-3)
        self.calibrated_with = len(good_parts)
        return self.threshold

    def set_sensitivity(self, sensitivity: float) -> None:
        base = self.threshold * max(self.sensitivity, 1e-3)
        self.sensitivity = float(np.clip(sensitivity, 0.5, 2.0))
        self.threshold = base / self.sensitivity

    # -- scoring ---------------------------------------------------------------------------
    def _prep(self, img: np.ndarray) -> np.ndarray:
        return cv2.resize(img, (self.size, self.size), interpolation=cv2.INTER_AREA)

    def _raw_map(self, feat: np.ndarray) -> np.ndarray:
        h, w, c = feat.shape
        d = nn_distance(feat.reshape(-1, c), self.bank, self.bank_sq)
        m = d.reshape(h, w)
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        return cv2.GaussianBlur(m, (0, 0), sigmaX=4)

    def score(self, img: np.ndarray, out_size: tuple[int, int] | None = None) -> ScoreResult:
        if self.bank is None:
            raise RuntimeError("Model not fitted: register a golden sample first")
        h0, w0 = img.shape[:2]
        feat = self.extractor([self._prep(img)])[0]
        raw = self._raw_map(feat)
        norm = raw / self.threshold
        norm = cv2.resize(norm, out_size or (w0, h0), interpolation=cv2.INTER_LINEAR)
        result = ScoreResult(score=float(norm.max()), raw_score=float(raw.max()), anomaly_map=norm)
        result.regions = find_regions(norm)
        return result

    # -- persistence -----------------------------------------------------------------------
    def state(self) -> dict:
        return {
            "bank": self.bank,
            "threshold": np.float32(self.threshold),
            "auto_threshold": np.float32(self.auto_threshold),
            "sensitivity": np.float32(self.sensitivity),
            "calibrated_with": np.int32(self.calibrated_with),
            "size": np.int32(self.size),
            "rotation_invariant": np.bool_(self.rotation_invariant),
            "golden": np.stack(self.golden),
        }

    def load_state(self, state) -> "GoldenSampleModel":
        self.bank = np.ascontiguousarray(state["bank"], dtype=np.float32)
        self.bank_sq = (self.bank * self.bank).sum(1)
        self.threshold = float(state["threshold"])
        self.auto_threshold = float(state["auto_threshold"])
        self.sensitivity = float(state["sensitivity"])
        self.calibrated_with = int(state["calibrated_with"])
        self.size = int(state["size"])
        self.rotation_invariant = bool(state["rotation_invariant"])
        self.golden = list(state["golden"])
        return self


def find_regions(norm_map: np.ndarray, level: float = 0.85, max_regions: int = 5) -> list[Region]:
    """Connected blobs above `level` (relative to the threshold), largest peak first."""
    if norm_map.max() < 1.0:
        return []
    mask = (norm_map >= level).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        peak = float(norm_map[labels == i].max())
        if peak < 1.0 or area < 12:
            continue
        regions.append(Region((int(x), int(y), int(x + w), int(y + h)), peak, int(area)))
    regions.sort(key=lambda r: r.peak, reverse=True)
    return regions[:max_regions]


def heatmap_overlay(img: np.ndarray, norm_map: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """RGB overlay: transparent below 60% of threshold, jet colour above."""
    m = np.clip(norm_map / 1.6, 0, 1)
    colored = cv2.applyColorMap((m * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    weight = (np.clip((norm_map - 0.6) / 0.6, 0, 1) * alpha)[..., None]
    return (img * (1 - weight) + colored * weight).astype(np.uint8)
