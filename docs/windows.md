# Windows

Version 0.4.2 includes a native Windows desktop app. The installer
includes Python and the SSH/SFTP component; WSL and a separate OpenSSH
installation are not required.

## Install and connect

1. Download and open the Windows installer from [v0.4.2](https://github.com/cuiqirui99/DFT-AGENT/releases/tag/v0.4.2).
2. Launch **DFT Agent** from the Start menu.
3. Under **Cluster setup**, enter the cluster's full login hostname, username
   and SSH port, then the Slurm settings and solver paths supplied by your institution.
4. Use your SSH key/agent, or enter the SSH password in the app. Click **Check
   cluster connection** before submitting a calculation.

Your cluster needs Linux, Slurm, Python 3 and an enabled SFTP subsystem.
VASP also needs a license and your own POTCAR library. For Quantum ESPRESSO,
use the [0.4.3 preview setup](qe.md); it does not require a VASP license.
The Windows app connects directly to that cluster. Local file paths use Windows locations; cluster paths continue
to use absolute Linux paths, such as `/scratch/your-user/dft-agent`.

## Authentication and host keys

The app uses keys from `%USERPROFILE%\.ssh\` and the Windows OpenSSH agent
or Pageant when available. A password is held only in memory and is not saved
in calculation files. An encrypted default private key can use the entered
password as its passphrase.

New server host keys are saved in `%USERPROFILE%\.ssh\known_hosts`, matching
the app's existing accept-new policy. A changed host key is rejected. Verify
a changed key with your cluster administrator before editing that file.

Enter the actual hostname and port in Cluster setup. The native Windows
connection does not read OpenSSH `Host` aliases, `ProxyJump`, or control
sockets. If your institution requires a VPN, connect it first. Interactive
MFA prompts that require a fresh response during login are not supported by
the unattended monitor; ask your institution about an approved SSH key or
agent setup. WSL remains an alternative for sites that require an existing
OpenSSH control connection.

## Files

Preferences and cluster settings are in `%USERPROFILE%\.dft-agent\`.
Runs are in `%USERPROFILE%\dft-agent-runs\`, unless an earlier
`vasp-slurm-agent-runs` folder already exists. Use File Explorer to copy,
inspect or back up these folders. `DFT_AGENT_HOME`, `DFT_AGENT_CONFIG` and
`DFT_AGENT_RUNS` can override their locations.

## Python installation

Developers can install into Python 3.11 or newer from PowerShell, starting
in the source folder:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install '.[agent]'
.\.venv\Scripts\dft-agent ui
```

The Python installation opens the interface in a browser. The desktop
installer includes its own window and does not require these commands.
