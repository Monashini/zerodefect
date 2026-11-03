"""ZeroDefect-Synth: a reproducible sanity benchmark on procedurally generated parts
(gear, brass washer, PCB-like bracket) x 5 defect types, with random pose, lighting and
sensor noise. One golden sample per part, zero defect images.

    python benchmark/synth_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from zerodefect.config import settings  # noqa: E402
from zerodefect.detector import FeatureExtractor, GoldenSampleModel  # noqa: E402
from zerodefect.synth import DEFECTS, PARTS, make_part  # noqa: E402


def run(n_good: int = 40, n_per_defect: int = 8, golden_seeds=(1, 2, 3), n_calib: int = 4):
    ext = FeatureExtractor(settings.backbone_path)
    rows = []
    for kind in PARTS:
        for gs in golden_seeds:
            golden, _ = make_part(kind, None, seed=gs)
            model = GoldenSampleModel(ext).fit([golden])
            test = [(make_part(kind, None, seed=10_000 + gs * 1000 + i)[0], None) for i in range(n_good)]
            test += [
                (make_part(kind, d, seed=20_000 + gs * 1000 + j * 100 + i)[0], d)
                for j, d in enumerate(DEFECTS)
                for i in range(n_per_defect)
            ]
            y = np.array([int(d is not None) for _, d in test])
            raw = np.array([model.score(img).raw_score for img, _ in test])
            auto_v = raw > model.threshold
            calib = [make_part(kind, None, seed=30_000 + gs * 100 + k)[0] for k in range(n_calib)]
            model.calibrate(calib)
            cal_v = raw > model.threshold
            per_defect = {
                d: float(np.mean([v for v, (_, dd) in zip(cal_v, test) if dd == d])) for d in DEFECTS
            }
            rows.append({
                "part": kind,
                "golden_seed": gs,
                "image_auroc": float(roc_auc_score(y, raw)),
                "auto": {"recall": float(auto_v[y == 1].mean()), "false_alarm": float(auto_v[y == 0].mean())},
                "calibrated": {"recall": float(cal_v[y == 1].mean()), "false_alarm": float(cal_v[y == 0].mean())},
                "recall_by_defect_calibrated": per_defect,
            })
            print(json.dumps(rows[-1]))
    return rows


def main():
    rows = run()
    out = Path(__file__).with_name("results_synth.json")
    out.write_text(json.dumps(rows, indent=2))
    print("\n| part | image AUROC | recall (auto / +4 OK parts) | false alarms (auto / +4 OK parts) |")
    print("|---|---|---|---|")
    for kind in PARTS:
        rs = [r for r in rows if r["part"] == kind]
        m = lambda f: np.mean([f(r) for r in rs])  # noqa: E731
        print(
            f"| {kind} | {m(lambda r: r['image_auroc']):.3f} | "
            f"{m(lambda r: r['auto']['recall']):.0%} / {m(lambda r: r['calibrated']['recall']):.0%} | "
            f"{m(lambda r: r['auto']['false_alarm']):.0%} / {m(lambda r: r['calibrated']['false_alarm']):.0%} |"
        )
    print(f"| **all** | **{np.mean([r['image_auroc'] for r in rows]):.3f}** | "
          f"{np.mean([r['auto']['recall'] for r in rows]):.0%} / {np.mean([r['calibrated']['recall'] for r in rows]):.0%} | "
          f"{np.mean([r['auto']['false_alarm'] for r in rows]):.0%} / {np.mean([r['calibrated']['false_alarm'] for r in rows]):.0%} |")
    print("\nRecall by defect type (calibrated):")
    for d in DEFECTS:
        print(f"  {d:14s} {np.mean([r['recall_by_defect_calibrated'][d] for r in rows]):.0%}")


if __name__ == "__main__":
    main()
