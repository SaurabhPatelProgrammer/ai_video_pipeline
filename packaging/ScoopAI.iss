#define MyAppName "Scoop AI"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Scoop AI"
#define MyAppExeName "ScoopAIClient.exe"

[Setup]
AppId={{9EE43436-42EC-49A5-91E4-DDA34E745B37}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={autopf}\Scoop AI
DefaultGroupName=Scoop AI
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installer
OutputBaseFilename=ScoopAI-Setup-{#MyAppVersion}
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\ScoopAIClient\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autodesktop}\Scoop AI"; Filename: "{app}\{#MyAppExeName}"
Name: "{group}\Scoop AI"; Filename: "{app}\{#MyAppExeName}"
Name: "{userstartup}\Scoop AI Background"; Filename: "{app}\{#MyAppExeName}"; Parameters: "--no-browser"; WorkingDir: "{app}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "Open Scoop AI setup"; Flags: nowait postinstall skipifsilent
