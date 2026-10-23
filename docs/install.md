# Install

Download the macOS or Windows installer from [v0.4.6](https://github.com/cuiqirui99/DFT-AGENT/releases/tag/v0.4.6).

## Desktop app

- **Apple Silicon Mac, macOS 14+:** open the `.dmg`, copy **DFT Agent** to Applications,
  then open it.
- **Windows x64:** run the `.exe` installer, then open **DFT Agent** from the
  Start menu. If Microsoft WebView2 is missing, installation needs an internet
  connection to install it. [Windows connection details](windows.md).

Python, scientific libraries and model SDKs are included. You do not need
Python, pip or a terminal to use the desktop app. Windows connects natively;
WSL is optional.

The installers do not have a publisher certificate; the Mac app is not notarized. For a
trusted Mac download, try opening it, then use **System Settings → Privacy &
Security → Open Anyway** if offered. [Apple's instructions](https://support.apple.com/102445).
Windows may also block an unsigned app; a managed PC may need approval from
your IT team. [Microsoft's guidance](https://learn.microsoft.com/en-us/windows/apps/package-and-deploy/smartscreen-reputation).

## First use

1. Create a project, then open **Connections → Model** to connect Codex CLI,
   Claude Code CLI or an API provider. The app tests the connection with a real
   request. The app can open the official CLI installer and sign-in. [Model setup](models.md).
2. Open **Cluster** and enter `ssh name@host`. Follow the password or login-code
   prompts that apply to your site, then review the detected software.
3. Return to chat, import a structure under **Files and runs → Files → Add files**,
   and describe the calculation. Review **Prepare inputs** before **Submit**.

For offline samples and manual preparation, open **Settings → Legacy workbench**.
The previous **Runs**, **New calculation** and **Cluster setup** pages remain there.

## Your first cluster setup

The **Cluster** guide asks for the SSH host and username, tries existing keys,
then requests a password or login code only when required. Passwords are saved
in the system keychain; codes are temporary. **Sign in through a terminal instead**
is available for unusual login prompts.

The authenticated session is reused for later commands and background workers
on macOS, Linux and Windows. Login codes are not saved. When a session expires,
return to **Connections → Cluster** to sign in again, then use **Reconnect**
on the affected task. On Windows, complete code sign-in inside the app;
a separate terminal session is not shared with its workers.

After connection, check the **Work folder** and **Scheduler**. Choose Slurm
or PBS explicitly when detection is ambiguous. The remote machine needs
Python, Bash and the selected scheduler's commands. Software launchers and
resource requests must match the cluster; use a working job script from
its documentation or support team.

| Setting | Where to get it |
| --- | --- |
| SSH host, user, port | Your cluster's login address and personal account. These are separate from your model account. |
| Work folder | Your writable scratch or project directory, as an absolute path on the cluster. |
| Partition or queue, account | Your cluster portal or a working job example. The account identifies the compute allocation. |
| Program command and module commands | The launch and environment lines from a working VASP or QE job. |
| Tasks, nodes, time | Resources allowed by your allocation and suitable for the calculation. |
| POTCAR or UPF folder | Your licensed VASP library or the directory holding the QE pseudopotentials. |

Detected environments appear after sign-in. Use **Connections → Software** to
add a missing environment. Its **Options → Additional settings (JSON)** accepts
site-specific settings such as `account`, `setup_commands` and `extra_sbatch`.
For example, `{"account": "my-allocation", "extra_sbatch": ["--mem=16G"]}`.
Use only values accepted by your site. Keep scheduler directives out of module
commands. The connection test does not submit a calculation or prove allocation
access; submission errors remain visible on the task.

### VASP potentials

Detection looks for your existing licensed POTCAR library. If it is missing,
enter its parent directory: use `/path/to/potpaw_PBE` for a library containing
`/path/to/potpaw_PBE/Si/POTCAR`. Ask your VASP administrator for this path.

The saved library is reused. Before submission, the app checks and joins the
required potentials in POSCAR element order on the cluster. Leave the element
mapping unset to use the defaults, or supply `potcar_symbols` in the software
settings when your library requires specific variants. POTCAR contents are
not sent to the model. Local folder/archive import is not supported.

### Install Quantum ESPRESSO

When QE is missing, the cluster guide offers **Install Quantum ESPRESSO**.
Enter the build partition and account, adjust module commands if needed, and
click **Plan the build**. Review the destination and resources, tick the
confirmation box, then **Start the build**. **Check the build** shows progress
or an error; a completed build adds its own QE environment.

The agent can plan installation, but cannot replace the app's confirmation.
Building requires a compatible MPI Fortran toolchain and source-download access.
Cluster restrictions or missing dependencies may need administrator help.
Public PBE UPFs can be installed for a task with **Install public UPFs**;
check that the selected potentials suit the calculation, especially SOC.
[QE methods and setup](qe.md).

## Python installation from source

For Linux users and developers, use Python 3.11 or newer. On macOS/Linux:

```bash
git clone --branch v0.4.6 https://github.com/cuiqirui99/DFT-AGENT.git dft-agent
cd dft-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[agent]'
dft-agent ui
```

These commands install version `0.4.6`.
The Python interface opens in a local browser at
`http://127.0.0.1:8501`; `--no-browser` skips opening it and `--port` selects
another port. OpenSSH is required for macOS/Linux cluster connections.

`[agent]` adds model SDKs and keychain support; omitting it still allows
manual preparation and calculation. `[dev]` adds test and build tools.

## Files and updates

| What | Location |
| --- | --- |
| Project index, conversations and connections | `~/.dft-agent/workspace.sqlite3` |
| Project files, runs and artifacts | `~/.dft-agent/projects/` |
| Preferences and desktop log | `~/.dft-agent/` |
| Legacy cluster settings | `~/.dft-agent/cluster.json` |
| Legacy runs | `~/dft-agent-runs/` |

On Windows, `~` means your user folder, normally `%USERPROFILE%`.
Older settings under `~/.config/vasp-slurm-agent/` are copied on first start;
an existing `~/vasp-slurm-agent-runs/` folder continues to be used.
`DFT_AGENT_HOME` changes the workspace location. `DFT_AGENT_CONFIG` and
`DFT_AGENT_RUNS` apply to the legacy configuration and run folders.

Close the app before installing a newer version. Keep settings and run
folders to retain your work. Each published version gets its own tag and
Release; earlier tags, releases and assets remain available and unchanged.

To remove the desktop app, move it to the Trash on macOS or use Installed
apps on Windows. Keep the folders above unless you also want to remove your
settings and results. Use **Connections → Saved connections → Manage connection
→ Revoke** to disable a connection and remove its stored credential before
uninstalling. Check any reported keychain cleanup error. Legacy remembered keys
have separate controls in the legacy workbench.
