import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def extractor():
    from zerodefect.config import settings
    from zerodefect.detector import FeatureExtractor

    if not settings.backbone_path.exists():
        pytest.skip("backbone missing: python scripts/download_model.py")
    return FeatureExtractor(settings.backbone_path)
