"""Zero-training benchmark on MVTec AD (the standard industrial anomaly-detection dataset).

Uses only k "good" training images per category as golden samples (default k=1) and no
defect images at all. Reports image-level AUROC, pixel-level AUROC and the accuracy of the
automatic pass/fail threshold.

Get the data from https://www.mvtec.com/company/research/datasets/mvtec-ad (free for
non-commercial research) and extract it, e.g. to ./mvtec_anomaly_detection/bottle/...

    python benchmark/mvtec_eval.py --root mvtec_anomaly_detection --shots 1
    python benchmark/mvtec_eval.py --root mvtec_anomaly_detection --categories bottle,screw --shots 1 2 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zerodefect.config import settings  # noqa: E402
from zerodefect.detector import FeatureExtractor, GoldenSampleModel  # noqa: E402

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]
# Parts that arrive in any orientation in MVTec: enable rotation-invariant golden banks.
ROTATING = {"screw", "metal_nut"}


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return img[:, :, ::-1].copy()


def eval_category(root: Path, cat: str, shots: int, extractor, size: int, pixel_res: int = 128, seed: int = 0):
    good_train = sorted((root / cat / "train" / "good").glob("*.png"))
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(good_train), shots, replace=False)
    goldens = [load_rgb(good_train[i]) for i in idx]
    model = GoldenSampleModel(extractor, size=size, rotation_invariant=cat in ROTATING)
    t0 = time.time()
    model.fit(goldens)
    fit_s = time.time() - t0

    labels, scores, verdicts, pix_true, pix_score = [], [], [], [], []
    t0 = time.time()
    n = 0
    for d in sorted((root / cat / "test").iterdir()):
        for p in sorted(d.glob("*.png")):
            img = load_rgb(p)
            res = model.score(img, out_size=(pixel_res, pixel_res))
            is_bad = d.name != "good"
            labels.append(int(is_bad))
            scores.append(res.score)
            verdicts.append(int(res.is_defect))
            if is_bad:
                gt_path = root / cat / "ground_truth" / d.name / (p.stem + "_mask.png")
                gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, (pixel_res, pixel_res), interpolation=cv2.INTER_NEAREST) > 0
            else:
                gt = np.zeros((pixel_res, pixel_res), bool)
            pix_true.append(gt.ravel())
            pix_score.append(res.anomaly_map.ravel())
            n += 1
    infer_ms = (time.time() - t0) / max(n, 1) * 1000
    labels, scores, verdicts = np.array(labels), np.array(scores), np.array(verdicts)
    return {
        "category": cat,
        "shots": shots,
        "image_auroc": round(float(roc_auc_score(labels, scores)), 4),
        "pixel_auroc": round(float(roc_auc_score(np.concatenate(pix_true), np.concatenate(pix_score))), 4),
        "auto_threshold_acc": round(float((labels == verdicts).mean()), 4),
        "recall_at_auto": round(float(verdicts[labels == 1].mean()), 4),
        "false_alarm_at_auto": round(float(verdicts[labels == 0].mean()), 4),
        "n_test": int(n),
        "fit_s": round(fit_s, 2),
        "ms_per_image": round(infer_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--categories", default=",".join(CATEGORIES))
    ap.add_argument("--shots", type=int, nargs="+", default=[1])
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out", type=Path, default=Path("benchmark/results_mvtec.json"))
    args = ap.parse_args()

    extractor = FeatureExtractor(settings.backbone_path, input_size=args.size)
    cats = [c for c in args.categories.split(",") if (args.root / c).exists()]
    if not cats:
        sys.exit(f"No MVTec categories found under {args.root}")
    results = []
    for shots in args.shots:
        for cat in cats:
            r = eval_category(args.root, cat, shots, extractor, args.size)
            results.append(r)
            print(json.dumps(r))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))

    print("\n| shots | category | image AUROC | pixel AUROC | auto-threshold acc | ms/img |")
    print("|---|---|---|---|---|---|")
    for r in results:
        print(f"| {r['shots']} | {r['category']} | {r['image_auroc']:.3f} | {r['pixel_auroc']:.3f} | "
              f"{r['auto_threshold_acc']:.3f} | {r['ms_per_image']} |")
    for shots in args.shots:
        rs = [r for r in results if r["shots"] == shots]
        print(f"| {shots} | **mean** | **{np.mean([r['image_auroc'] for r in rs]):.3f}** | "
              f"**{np.mean([r['pixel_auroc'] for r in rs]):.3f}** | {np.mean([r['auto_threshold_acc'] for r in rs]):.3f} | |")


if __name__ == "__main__":
    main()
