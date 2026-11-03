import numpy as np
from sklearn.metrics import roc_auc_score

from zerodefect.detector import GoldenSampleModel, find_regions
from zerodefect.synth import DEFECTS, make_part


def test_one_shot_separates_good_from_defective(extractor):
    golden, _ = make_part("gear", None, seed=1)
    model = GoldenSampleModel(extractor).fit([golden])
    good = [model.score(make_part("gear", None, seed=100 + i)[0]).raw_score for i in range(12)]
    bad = [model.score(make_part("gear", d, seed=200 + i)[0]).raw_score for i, d in enumerate(DEFECTS * 2)]
    auroc = roc_auc_score([0] * len(good) + [1] * len(bad), good + bad)
    assert auroc > 0.9, auroc


def test_heatmap_points_at_the_defect(extractor):
    golden, _ = make_part("washer", None, seed=1)
    model = GoldenSampleModel(extractor).fit([golden])
    hits = 0
    for i in range(6):
        img, gt = make_part("washer", "stain", seed=400 + i)
        res = model.score(img)
        y, x = np.unravel_index(np.argmax(res.anomaly_map), res.anomaly_map.shape)
        ys, xs = np.nonzero(gt)
        dist = np.min(np.hypot(ys - y, xs - x))
        hits += dist < 25
    assert hits >= 5


def test_calibration_and_state_roundtrip(extractor):
    golden, _ = make_part("bracket", None, seed=2)
    model = GoldenSampleModel(extractor).fit([golden])
    thr = model.calibrate([make_part("bracket", None, seed=50 + i)[0] for i in range(3)])
    assert thr > 0 and model.calibrated_with == 3
    clone = GoldenSampleModel(extractor).load_state(model.state())
    img, _ = make_part("bracket", "scratch", seed=9)
    assert abs(clone.score(img).score - model.score(img).score) < 1e-4
    model.set_sensitivity(2.0)
    assert model.score(img).score > clone.score(img).score


def test_find_regions_empty_below_threshold():
    assert find_regions(np.full((64, 64), 0.5, np.float32)) == []
    m = np.zeros((64, 64), np.float32)
    m[10:20, 30:40] = 1.5
    r = find_regions(m)
    assert len(r) == 1 and r[0].box == (30, 10, 40, 20)
