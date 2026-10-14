# Install

Download the macOS or Windows installer from [v0.4.3](https://github.com/cuiqirui99/DFT-AGENT/releases/tag/v0.4.3).

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

1. Open **Runs → Load sample runs** to view completed results and plots offline.
2. Choose **VASP**. In **New calculation**, choose an example structure and **Manual**, then
   **Prepare inputs**. This needs neither a cluster nor a model account.
3. To submit a calculation, save your own **Cluster setup**: SSH access to a
   Linux Slurm cluster with Python 3 and the selected solver. VASP requires
   licensed VASP and POTCAR files. The [QE route](qe.md) can install QE and public UPFs.
4. For **Plan**, **Explain results** or model-assisted edits, configure your
   own model account under **Model**. Network access and the provider's billing
   apply. Bundled SDKs include no API keys or credits; Codex CLI is optional
   and installed separately. [Model setup](models.md).

The interactive structure viewer downloads 3Dmol.js. For offline use, turn
it off under **Preferences** to use the static plot. Automatic update checks
can also be turned off there.

## Your first cluster setup

You do not need to create `cluster.json` yourself. Leave **Configuration
file** at its default, fill **Cluster setup**, then click **Save cluster
settings**. For QE, follow [QE setup](qe.md). The settings below are for VASP.
Ask your institution's cluster support for a working VASP job
example and the following details:

| App field | Where to get the value |
| --- | --- |
| SSH host, user, port | The login hostname and your cluster login account; usually port 22. These are separate from a model API account. |
| Remote run folder | Your writable scratch or project directory on the cluster, as an absolute Linux path. This is not a folder on your laptop. |
| Remote POTCAR folder | The licensed potential directory supplied by your VASP administrator, containing element folders such as `Si/POTCAR`. |
| VASP command and environment setup | Copy the launch command and module/setup lines from your site's working VASP job. `srun vasp_std` is only a starting value. |
| SOC / noncollinear command | The site's launch command for `vasp_ncl`, required when using SOC or noncollinear spins. |
| Slurm partition, account, nodes, MPI tasks, time | Use the resources and project allocation in that job example or your cluster portal. The Slurm account is the project charged for the job, not the SSH username. |

After entering SSH details, **Detect from cluster** can suggest the remaining
values. Enter an SSH password in the sidebar if your site uses passwords;
otherwise use its configured SSH key/agent. Review the suggestions, save,
then click **Check cluster connection**. Neither action submits a calculation.
Leave optional mappings and extra options unchanged unless your site requires
them. A preset does not grant cluster access or a VASP licence.

### POTCAR

1. Click **Detect from cluster** to look for your licensed library.
2. Check **Remote POTCAR folder**. Enter the parent of the element folders:
   for `/path/to/potpaw_PBE/Si/POTCAR`, use `/path/to/potpaw_PBE`.
   If detection finds nothing, ask your VASP administrator for this path.
3. Leave **POTCAR element mapping** as `{}` to use the recommended potentials.
   Override it only when your library uses different variants, such as `{"Ti": "Ti_pv"}`.
4. Click **Save cluster settings**, then **Check cluster connection**.

The saved path is reused. Before submission, the app checks that the required
files exist and are nonempty, then joins them in POSCAR element order on the
cluster. The connection check alone does not validate every potential.
Local folder and archive import is not supported yet.

## Python installation from source

For Linux users and developers, use Python 3.11 or newer. On macOS/Linux:

```bash
git clone --branch v0.4.3 https://github.com/cuiqirui99/DFT-AGENT.git dft-agent
cd dft-agent
python3 -m venv .venv
source .venv/bin/activate
python -m pip install '.[agent]'
dft-agent ui
```

These commands install version `0.4.3`.
The Python interface opens in a local browser at
`http://127.0.0.1:8501`; `--no-browser` skips opening it and `--port` selects
another port. OpenSSH is required for macOS/Linux cluster connections.

`[agent]` adds model SDKs and keychain support; omitting it still allows
manual preparation and calculation. `[dev]` adds test and build tools.

## Files and updates

| What | Location |
| --- | --- |
| Cluster settings | `~/.dft-agent/cluster.json` |
| Preferences and desktop log | `~/.dft-agent/` |
| Runs | `~/dft-agent-runs/` |

On Windows, `~` means your user folder, normally `%USERPROFILE%`.
Older settings under `~/.config/vasp-slurm-agent/` are copied on first start;
an existing `~/vasp-slurm-agent-runs/` folder continues to be used.
`DFT_AGENT_HOME`, `DFT_AGENT_CONFIG` and `DFT_AGENT_RUNS` override these paths.

Close the app before installing a newer version. Keep settings and run
folders to retain your work. Each published version gets its own tag and
Release; earlier tags, releases and assets remain available and unchanged.

To remove the desktop app, move it to the Trash on macOS or use Installed
apps on Windows. Keep the folders above unless you also want to remove your
settings and results. Remembered keys remain in the system keychain; untick
**Remember on this computer** for each provider to remove them before uninstalling.
