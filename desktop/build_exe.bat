@echo off
REM Build ModeTrains.exe with PyInstaller (run from repo root).
REM Requires: pip install pyinstaller
REM Copyright (c) 2026 salim-slimani. MIT license.
pyinstaller --noconfirm --onefile --windowed --name ModeTrains ^
  --distpath dist desktop\modetrains_desktop.py
echo Built: dist\ModeTrains.exe
