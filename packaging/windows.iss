#ifndef AppVersion
  #error Pass /DAppVersion with the version being packaged.
#endif

[Setup]
AppId=DFTAgent
AppName=DFT Agent
AppVersion={#AppVersion}
AppPublisher=Qirui Cui
AppPublisherURL=https://github.com/cuiqirui99/dft-agent
DefaultDirName={localappdata}\Programs\DFT Agent
DefaultGroupName=DFT Agent
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist\installers
OutputBaseFilename=DFT-Agent-{#AppVersion}-Windows-x64
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\DFT Agent.exe
CloseApplications=yes
LicenseFile=..\LICENSE

[Files]
Source: "..\dist\MicrosoftEdgeWebview2Setup.exe"; Flags: dontcopy
Source: "..\dist\DFT Agent\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\DFT Agent"; Filename: "{app}\DFT Agent.exe"
Name: "{autodesktop}\DFT Agent"; Filename: "{app}\DFT Agent.exe"; Tasks: desktopicon

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; Flags: unchecked

[Run]
Filename: "{app}\DFT Agent.exe"; Description: "Open DFT Agent"; Flags: nowait postinstall skipifsilent

[Code]
function NeedsWebView2: Boolean;
var
  Version: String;
begin
  Result := not ((RegQueryStringValue(HKCU, 'Software\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')) or
    (RegQueryStringValue(HKLM32, 'SOFTWARE\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}', 'pv', Version) and (Version <> '') and (Version <> '0.0.0.0')));
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
  Recovery: String;
begin
  Result := '';
  if not NeedsWebView2 then
    Exit;

  Recovery := #13#10#13#10 + 'Check your internet connection and proxy settings, then retry. ' +
    'For offline installation, install the Microsoft WebView2 Evergreen Standalone Runtime (x64) from ' +
    'https://developer.microsoft.com/microsoft-edge/webview2/ and run this installer again.';
  try
    Log('Installing Microsoft WebView2 (internet required).');
    ExtractTemporaryFile('MicrosoftEdgeWebview2Setup.exe');
    if not Exec(ExpandConstant('{tmp}\MicrosoftEdgeWebview2Setup.exe'), '/silent /install', '',
      SW_SHOWNORMAL, ewWaitUntilTerminated, ResultCode) then
      Result := 'Could not start the Microsoft WebView2 installer: ' + SysErrorMessage(ResultCode) + Recovery
    else if ResultCode <> 0 then
      Result := 'Microsoft WebView2 installation failed (exit code ' + IntToStr(ResultCode) + ').' + Recovery
    else if NeedsWebView2 then
      Result := 'Microsoft WebView2 is still unavailable after installation.' + Recovery;
  except
    Result := 'Could not prepare Microsoft WebView2: ' + GetExceptionMessage + Recovery;
  end;

  if Result <> '' then
    Log(Result);
end;
