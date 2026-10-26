# Desktop builds

Version `0.4.8` provides installers for Apple Silicon Macs running macOS 14
or later and Windows x64. Python and the scientific/model SDKs are
bundled; VASP, POTCAR files, cluster time, API credentials and model credits
are not. Download the installers from [v0.4.8](https://github.com/cuiqirui99/DFT-AGENT/releases/tag/v0.4.8).
Codex CLI and Claude Code are optional external tools; install and sign in to
either one, or use an API connection in the app.

## Build from a checkout

Use a separate virtual environment on the target operating system and architecture:

```bash
python -m pip install '.[desktop,dev]' 'pyinstaller>=6.15,<7' 'ase>=3.23,<4'
python -m PyInstaller --noconfirm packaging/desktop.spec
```

On macOS, this produces `dist/DFT Agent.app`. Create a disk image after the
app has passed its checks:

```bash
mkdir -p dist/dmg
cp -R 'dist/DFT Agent.app' dist/dmg/
ln -s /Applications dist/dmg/Applications
hdiutil create -volname 'DFT Agent' -srcfolder dist/dmg -ov -format UDZO dist/DFT-Agent-0.4.8-macOS-arm64.dmg
```

On Windows, the app folder is `dist/DFT Agent`. The
[Windows packaging workflow](../.github/workflows/desktop.yml) downloads and
checks Microsoft's WebView2 installer, invokes Inno Setup with
[windows.iss](../packaging/windows.iss), and checks a fresh installation.
The resulting installer goes into `dist/installers/`. Run this on Windows;
a macOS build is not a Windows installation test.

## Check the installed copy

For example, after copying the Mac app into Applications:

```bash
python packaging/check_installed.py '/Applications/DFT Agent.app/Contents/MacOS/DFT Agent' evidence/macos
```

The same command accepts the installed Windows `DFT Agent.exe` path. It
creates isolated settings and runs, checks project creation and persistence,
the agent's MCP subprocess, the local SSH session service and native keychain,
bundled resources, sample copies, local input
preparation and cluster attachment, then starts the packaged local server and
native window. It records `bundle-check.json`, `server.log` and `window-check.json`. This requires
no model account or cluster and does not send a job.

Also open the installed app in a fresh user workspace and verify sample
viewing, manual preparation, language switching, configuration editing,
file selection and downloads. Keep the installer hash, commands, reports
and screenshots with the acceptance record. Passing unit tests or the
local-server check alone does not verify the native window or every user action.

Live SSH/solver calculations and live model calls require separate evidence.
The [recorded Claude checks](validation-0.4.1.md) used the actual SDK with
simulated HTTP responses; no live Claude API request was made. Code signing,
notarization, Windows installer execution and interactive checks should only
be marked complete when their own records exist.

Publish each completed version under a new tag and GitHub Release. Keep
previous release records and assets unchanged.
