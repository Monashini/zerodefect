"""Download the frozen ImageNet backbone (ResNet-18, ONNX model zoo, Apache-2.0, ~45 MB)."""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://media.githubusercontent.com/media/onnx/models/main/validated/vision/classification/resnet/model/resnet18-v1-7.onnx"
DEST = Path(__file__).resolve().parent.parent / "models" / "resnet18-v1-7.onnx"


def main():
    if DEST.exists() and DEST.stat().st_size > 40_000_000:
        print(f"Backbone already present: {DEST}")
        return
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}\n -> {DEST}")
    tmp = DEST.with_suffix(".part")
    urllib.request.urlretrieve(URL, tmp)
    if tmp.stat().st_size < 40_000_000:
        sys.exit("Download looks incomplete; please retry.")
    tmp.rename(DEST)
    print("sha256", hashlib.sha256(DEST.read_bytes()).hexdigest()[:16], "OK")


if __name__ == "__main__":
    main()
