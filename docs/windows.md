# Windows

The Windows x64 installer includes Python and native SSH/SFTP. WSL and a
separate OpenSSH installation are not required.

## Install and connect

1. Install [v0.4.5](https://github.com/cuiqirui99/DFT-AGENT/releases/tag/v0.4.5)
   and open **DFT Agent** from the Start menu.
2. Create a project. In **Connections → Model**, connect your CLI or API account.
3. In **Connections → Cluster**, enter `ssh name@host` and follow the sign-in
   prompts. The app tries your keys, then asks for a password or login code
   when needed.
4. Review the detected software and work folder. Use **Software** to add a
   solver or change its launcher, partition or queue, and resources.
5. Return to chat, import a structure under **Files and runs → Files**, and describe
   your calculation. Review the inputs before submitting.

Your cluster needs Linux, Slurm or PBS, Python 3 and SFTP. VASP needs a license
and your own POTCAR library. QE can use public pseudopotentials and the
[guided installation](install.md#your-first-cluster-setup).
Local paths use Windows locations; cluster paths use absolute Linux paths,
such as `/scratch/your-user/dft-agent`.

## Authentication and host keys

The app uses keys from `%USERPROFILE%\.ssh\` and the Windows OpenSSH agent
or Pageant when available. Saved SSH passwords stay in the system keychain.
Login codes are temporary. An encrypted private key can use the entered
password as its passphrase.

After password-plus-code or key-plus-code sign-in, the app keeps the
authenticated SSH session available to background workers. If it disconnects
or expires, sign in again under **Connections → Cluster**, then use
**Reconnect** on the affected task. Codes are not saved or replayed, and
reconnecting does not submit a new calculation. A separate terminal's SSH
session is not shared with the app.

The native connection reads `%USERPROFILE%\.ssh\config`, including hostname
aliases and jump-host settings. Enter the correct username and port in the
app. Connect your institution's VPN first if required. Site-specific prompts
beyond password and verification code may need your cluster support.

New server keys are saved in `%USERPROFILE%\.ssh\known_hosts`. A changed key
is rejected; verify it with your cluster administrator before editing that file.

## Files

Projects and connection records are under `%USERPROFILE%\.dft-agent\`.
The legacy workbench keeps its settings and runs separately. See
[files and updates](install.md#files-and-updates) before moving or removing them.

## Python installation

Developers can install into Python 3.11 or newer from PowerShell, starting
in the source folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install '.[agent]'
.\.venv\Scripts\dft-agent ui
```

This opens the interface in a browser. The desktop installer includes its own
window and does not require these commands. Codex CLI and Claude Code CLI,
if chosen, remain separate installations.
