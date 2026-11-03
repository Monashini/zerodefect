@echo off
cd /d %~dp0
python -m pip install -q -r requirements.txt
python scripts\download_model.py
echo ZeroDefect running at http://localhost:8000
python -m uvicorn zerodefect.server:app --host 0.0.0.0 --port 8000
