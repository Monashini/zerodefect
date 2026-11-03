#!/usr/bin/env bash
# One-command start: installs deps, fetches the backbone, opens http://localhost:8000
set -e
cd "$(dirname "$0")"
python3 -m pip install -q -r requirements.txt
python3 scripts/download_model.py
echo "ZeroDefect running at http://localhost:8000"
python3 -m uvicorn zerodefect.server:app --host 0.0.0.0 --port 8000
