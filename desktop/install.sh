#!/usr/bin/env bash
# ModeTrains Desktop installer (Linux / macOS).
# Creates a venv, installs the library, adds a .desktop launcher (Linux).
#
#   bash desktop/install.sh
#
# Copyright (c) 2026 salim-slimani. MIT license.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.venv-desktop"

echo "== ModeTrains Desktop setup =="
python3 -m venv "$VENV"
"$VENV/bin/pip" install --upgrade pip
"$VENV/bin/pip" install -r "$ROOT/requirements.txt"
"$VENV/bin/pip" install -e "$ROOT"

if [ -d "${HOME}/.local/share/applications" ]; then
  cat > "${HOME}/.local/share/applications/modetrains.desktop" <<EOF
[Desktop Entry]
Name=ModeTrains Desktop
Comment=Fast, memory-efficient LLM fine-tuning
Exec=$VENV/bin/python $ROOT/desktop/modetrains_desktop.py
Path=$ROOT
Terminal=false
Type=Application
Categories=Development;Science;
EOF
  echo "Launcher installed: ~/.local/share/applications/modetrains.desktop"
fi
echo "Done. Run: $VENV/bin/python $ROOT/desktop/modetrains_desktop.py"
