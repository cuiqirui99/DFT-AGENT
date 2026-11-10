# Install

Download the macOS installer from [v0.5.0](https://github.com/cuiqirui99/DFT-AGENT/releases/tag/v0.5.0).

## Desktop app

For **Apple Silicon Macs with macOS 14 or later**: open the `.dmg`, drag
**DFT Agent** to Applications, then launch it. Python, scientific libraries
and model SDKs are included.

The app is signed ad hoc and not notarized, so macOS stops the first launch
of a downloaded copy. For a download you trust, try to open the app once,
then open **System Settings → Privacy & Security**, click **Open Anyway** and
confirm. See [Apple's instructions](https://support.apple.com/102445).
A managed Mac may require your IT team's approval.

Version 0.5.0 does not include a Windows installer or a Python package.
[Earlier releases](https://github.com/cuiqirui99/DFT-AGENT/releases) remain available.

## First use

1. Create a project, then open **Connections → Model** to connect Codex CLI,
   Claude Code CLI or an API provider. The app tests the connection with a real
   request. The app can open the official CLI installer and sign-in. [Model setup](models.md).
2. Open **Cluster** and enter `ssh name@host`. Follow the password or login-code
   prompts that apply to your site. The app then sets up the cluster's software;
   follow it in **Connections → Software**.
3. Return to chat, import a structure under **Structures and runs → Structures →
   Add a structure → Upload**, and describe the calculation. Review **Prepare inputs**
   before **Submit**.

For offline samples and manual preparation, open **Settings → Classic workbench**.
The previous **Runs**, **New calculation** and **Cluster setup** pages remain there.

## Your first cluster setup

The **Cluster** guide asks for the SSH host and username, tries existing keys,
then requests a password or login code only when required. Passwords are saved
in the system keychain; codes are temporary. **Sign in through a terminal instead**
is available for unusual login prompts.

The authenticated session is reused for later commands and background workers.
Login codes are not saved. When a session expires, return to **Connections →
Cluster** to sign in again, then use **Reconnect** on the affected task.

After the sign-in the app sets the cluster up by itself. It probes the cluster
read-only: scheduler commands and their versions, partitions, the module
listing, compilers, MPI, glibc and storage. From that output the connected model
judges which scheduler submits jobs, which partition and account to use, and
which module or program provides each code. The app then checks the judgement.
It runs the chosen module loads and version commands on the login node, submits
a short test job for VASP, Quantum ESPRESSO, ABINIT, CP2K, Elk, SIESTA, LAMMPS
and UppASD and reads its state, and checks Python, Wannier90 and RSPt on the
login node. Only a passed test or check marks a program **Verified** in
**Connections → Software**. The scheduler judgement counts as confirmed once the
first test job has finished. With a connected model, a Slurm cluster that ships
PBS-style wrappers (qsub, qstat, qdel beside sbatch) needs no manual choice.
Notes you write under **Notes about this cluster for the model** are read at the
next check.

The automatic setup starts only within an hour of a sign-in or a successful
connection test. For a cluster saved by an earlier version,
**Connections → Software** asks you to press **Test connection** first.

Missing or broken open-source programs are offered for installation one by
one; tick **Install missing open-source software automatically** to let the
app install them without asking. Automatic installation needs a storage quota
that the app can read and enough room under it; otherwise the program stays
**Not installed** with a note. **Install** under **Details** works without a
readable quota, using the file system's free space, but not when the space
read is too small or no free space could be read. A failed installation is
not started again automatically for a week unless its inputs change;
**Install** starts it at once.

This version cannot submit jobs to LSF or SGE. When the model finds one of them,
the automatic setup stops and says so. If Slurm or PBS also runs on such a
cluster, choose it under **Connections → Cluster → Scheduler** and add the
software under **Manual setup**. Without a connected model, or when the model
does not answer, the built-in rules detect the scheduler. When they find both
Slurm and PBS commands, as on a Slurm cluster with PBS-style wrappers, choose
the scheduler there yourself. A cluster set up by the built-in rules is checked
again with the model once, after a later model connection test succeeds;
connecting a model includes this test. This check also needs a sign-in or a
successful connection test of the cluster within the last hour; otherwise
**Connections → Software** asks you to press **Test connection**. Installing a
program or turning on automatic installation afterwards does not repeat it.
The remote machine needs Python, Bash and the scheduler's commands.

| Setting | Where to get it |
| --- | --- |
| SSH host, user, port | Your cluster's login address and personal account. These are separate from your model account. |
| Work folder | Your writable scratch or project directory, as an absolute path on the cluster. |
| Partition or queue, account | Your cluster portal or a working job example. The account identifies the compute allocation. |
| Program command and module commands | The launch and environment lines from a working VASP or QE job. |
| Tasks, nodes, time | Resources allowed by your allocation and suitable for the calculation. |
| POTCAR or UPF folder | Your licensed VASP library or the directory holding the QE pseudopotentials. |

The verified environments appear after the check. Use **Manual setup** under
**Connections → Software** to add or change an environment yourself. Its
**Options → Additional settings (JSON)** accepts site-specific settings such as
`account`, `setup_commands` and `extra_sbatch`.
For example, `{"account": "my-allocation", "extra_sbatch": ["--mem=16G"]}`.
Use only values accepted by your site. Keep scheduler directives out of module
commands. The connection test itself submits no job and does not prove
allocation access; a passed test job shows that its partition and account
accept jobs. Submission errors remain visible on the task.

### VASP potentials

Detection looks for your existing licensed POTCAR library. If it is missing,
enter its parent directory under **Connections → Software**: use
`/path/to/potpaw_PBE` for a library containing `/path/to/potpaw_PBE/Si/POTCAR`.
Ask your VASP administrator for this path.

The saved library is reused. Before submission, the app checks and joins the
required potentials in POSCAR element order on the cluster. Leave the element
mapping unset to use the defaults, or supply `potcar_symbols` in the software
settings when your library requires specific variants. POTCAR contents are
not sent to the model. Local folder/archive import is not supported.

### Install Quantum ESPRESSO

When QE is missing, **Connections → Software** offers **Install** for it, unless
automatic installation is on. The app then runs one installation job in your
cluster account and storage, followed by a small test. ABINIT, CP2K, Elk,
SIESTA, LAMMPS and UppASD are offered the same way.

If the app's own installation does not suit your cluster, open **Connections →
Cluster → Advanced installation**. While QE is missing, it shows **Install Quantum
ESPRESSO**. Enter the build partition and account, adjust module commands if
needed, and click **Plan the build**. Review the destination and resources, tick
the confirmation box, then **Start the build**. **Check the build** shows progress
or an error; a completed build adds its own QE environment.

In any managed installer, use **Scheduler options** for required memory, QoS or constraints, one option per line.

The agent can plan an installation. It can start one only while you have turned
on automatic installation. Building QE requires a compatible MPI Fortran
toolchain and source-download access.
Cluster restrictions or missing dependencies may need administrator help.
Public PBE UPFs can be installed for a task with **Install public UPFs**;
check that the selected potentials suit the calculation, especially SOC.
[QE methods and setup](qe.md).

## Command line

The Mac app includes the CLI used in the examples. In Terminal, define this
shortcut for the current shell:

```bash
dft-agent() { "/Applications/DFT Agent.app/Contents/MacOS/DFT Agent" --cli "$@"; }
dft-agent --help
```

Adjust the path if you installed the app elsewhere. No Python installation is needed.
[CLI examples](examples.md).

## Files and updates

| What | Location |
| --- | --- |
| Project index, conversations and connections | `~/.dft-agent/workspace.sqlite3` |
| Project files, runs and artifacts | `~/.dft-agent/projects/` |
| Preferences and desktop log | `~/.dft-agent/` |
| Legacy cluster settings | `~/.dft-agent/cluster.json` |
| Legacy runs | `~/dft-agent-runs/` |

Older settings under `~/.config/vasp-slurm-agent/` are copied on first start;
an existing `~/vasp-slurm-agent-runs/` folder continues to be used.
`DFT_AGENT_HOME` changes the workspace location. `DFT_AGENT_CONFIG` and
`DFT_AGENT_RUNS` apply to the legacy configuration and run folders.

Close the app before installing a newer version. Keep settings and run
folders to retain your work. Opening an existing workspace with 0.5.0 upgrades
it; versions 0.4.4 to 0.4.9 then report "This workspace requires a newer DFT
Agent version." and cannot open it. Copy `~/.dft-agent` first if you may want to
go back. Each published version gets its own tag and Release; earlier tags,
releases and assets remain available and unchanged.

To remove the desktop app, move it to the Trash. Keep the folders above unless
you also want to remove your settings and results. Use **Connections → Saved connections → Manage connection
→ Revoke** to disable a connection and remove its stored credential before
uninstalling. Check any reported keychain cleanup error. Legacy remembered keys
have separate controls in the legacy workbench.
