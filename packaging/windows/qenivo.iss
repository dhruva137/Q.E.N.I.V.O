; Inno Setup script: QENIVO-Setup-0.1.0.exe from the PyInstaller folder dist\QENIVO.
; Per-user install (no administrator rights, no system or security settings changed):
; files go to %LOCALAPPDATA%\Programs\QENIVO, shortcuts to the user's Start menu.
#define AppVersion "0.1.0"

[Setup]
AppId={{8F456FF3-320F-4473-8E8E-37CBCEF6715E}
AppName=QENIVO
AppVersion={#AppVersion}
AppPublisher=Team Epoch Zero
DefaultDirName={localappdata}\Programs\QENIVO
DefaultGroupName=QENIVO
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputDir=..\..\dist
OutputBaseFilename=QENIVO-Setup-{#AppVersion}
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\..\LICENSE
UninstallDisplayName=QENIVO {#AppVersion}

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Files]
Source: "..\..\dist\QENIVO\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\QENIVO"; Filename: "{app}\QENIVO.exe"
Name: "{group}\Uninstall QENIVO"; Filename: "{uninstallexe}"
Name: "{userdesktop}\QENIVO"; Filename: "{app}\QENIVO.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\QENIVO.exe"; Description: "Start QENIVO"; Flags: nowait postinstall skipifsilent
