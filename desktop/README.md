# ModeTrains Desktop

Graphical front-end for the `modetrains` library. Standard library only
(`tkinter`) — no extra dependencies.

Copyright (c) 2026 salim-slimani. MIT license.

## Run without installing

```bash
python desktop/modetrains_desktop.py
```

Three tabs: **Train** (form + live log), **Infer** (prompt + output),
**Hardware** (device info + VRAM estimate).

## Install (Windows)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\desktop\install.ps1
```

Creates an isolated `.venv-desktop`, installs the library in editable mode,
and adds Start Menu + Desktop shortcuts.

## Install (Linux / macOS)

```bash
bash desktop/install.sh
```

## Portable .exe (no install needed)

Download `ModeTrains.exe` from
[GitHub Releases](https://github.com/salim-studio/modetrains/releases)
and double-click. Or build it yourself:

```bat
pip install pyinstaller
desktop\build_exe.bat
```

## Full installer (maintainers)

1. Build `dist\ModeTrains.exe` (above).
2. Install Inno Setup 6 and run `iscc desktop\windows-installer.iss`.
3. Ship `ModeTrains-Setup-0.2.1.exe`.

## Smoke test (headless, CI-friendly)

```bash
python desktop/modetrains_desktop.py --self-test
python tests/test_desktop.py
```
