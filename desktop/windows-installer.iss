; Inno Setup script — builds ModeTrains-Setup.exe from dist\ModeTrains.exe.
; Requires Inno Setup 6 (https://jrsoftware.org/isinfo.php), then:
;   iscc desktop\windows-installer.iss
; Copyright (c) 2026 salim-slimani. MIT license.
#define MyAppName "ModeTrains Desktop"
#define MyAppVersion "0.2.0"
#define MyAppPublisher "salim-slimani"

[Setup]
AppId={{8F3A1C2B-4D5E-4F60-A1B2-C3D4E5F60718}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\ModeTrains
DefaultGroupName=ModeTrains
OutputBaseFilename=ModeTrains-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64

[Files]
Source: "..\dist\ModeTrains.exe"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{group}\ModeTrains Desktop"; Filename: "{app}\ModeTrains.exe"
Name: "{autodesktop}\ModeTrains Desktop"; Filename: "{app}\ModeTrains.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a &desktop icon"

[Run]
Filename: "{app}\ModeTrains.exe"; Description: "Launch ModeTrains Desktop"; Flags: nowait postinstall skipifsilent
