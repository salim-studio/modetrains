#Requires -Version 5.1
<#
  ModeTrains Desktop installer (Windows).
  Creates a virtual environment, installs the library, and adds
  Start Menu + Desktop shortcuts.

  Run (PowerShell):
    Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
    .\desktop\install.ps1

  Copyright (c) 2026 salim-slimani. MIT license.
#>
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Venv = Join-Path $Root ".venv-desktop"
$Py = Join-Path $Venv "Scripts\python.exe"

Write-Host "== ModeTrains Desktop setup ==" -ForegroundColor Cyan

if (-not (Test-Path $Py)) {
  Write-Host "Creating virtual environment..." -ForegroundColor Yellow
  python -m venv $Venv
}
& $Py -m pip install --upgrade pip
& $Py -m pip install -r (Join-Path $Root "requirements.txt")
& $Py -m pip install -e $Root

$Wsh = New-Object -ComObject WScript.Shell
$Target = "`"$Py`" `"$Root\desktop\modetrains_desktop.py`""
$Icon = Join-Path $Root "assets\logo.svg"
foreach ($dir in @(
  (Join-Path ([Environment]::GetFolderPath("Programs")) "ModeTrains"),
  ([Environment]::GetFolderPath("Desktop")))) {
  if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  $lnk = $Wsh.CreateShortcut((Join-Path $dir "ModeTrains Desktop.lnk"))
  $lnk.TargetPath = $Py
  $lnk.Arguments = "`"$Root\desktop\modetrains_desktop.py`""
  $lnk.WorkingDirectory = $Root
  $lnk.Description = "ModeTrains Desktop — fast LLM fine-tuning"
  $lnk.Save()
  Write-Host "Shortcut created: $dir" -ForegroundColor Green
}

Write-Host "Done. Launch 'ModeTrains Desktop' from the Start Menu." -ForegroundColor Green
