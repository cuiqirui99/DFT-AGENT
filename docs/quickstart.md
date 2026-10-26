# Quickstart

## 1. Install and create a project

Download [v0.4.8](https://github.com/cuiqirui99/DFT-AGENT/releases/tag/v0.4.8),
install the Mac `.dmg` or Windows `.exe`, and open **DFT Agent**.
Python and model SDKs are included. Linux and source installations are covered
in [Install](install.md).

Click **Create your first project**, enter a name, then **Create project**.
Each project has its own conversations, files and calculations. Use
**New project** for another study and **New conversation** for a new chat
within the selected project.

## 2. Connect a model

Open **Connections → Model**:

- **Codex CLI or Claude Code CLI:** uses a ChatGPT or Claude plan you already
  have. Each card shows its next step; **Install and sign in** opens a terminal
  window that installs the CLI and starts its sign-in in your browser. When you
  have signed in, the card offers **Connect**.
- **API:** choose a provider, paste its API key and click **Connect**. The app
  checks the region and available models. **Compatible API** also needs the
  address and model name from your provider.
- **Local model:** start a compatible local server, select it and click **Connect**.
  Use **Other local server** for a custom address or port.

The connection test makes a real model request. Saved connections can be
checked again with **Test connection**. API keys are stored in the system
keychain. Do not enter them in chat. [Account and token guide](models.md).

Codex and Claude CLI connections offer **Automatic** mode, which lets the CLI use
project tools across several steps. **Step by step** uses checked actions one
message at a time and is also used by API providers. Both require input review
before a calculation is submitted.

## 3. Connect your cluster and software

In **Connections → Cluster**, enter your SSH address, for example
`ssh name@cluster.example.org`. If you enter only a host, the app asks for
your username. SSH keys are tried first; a secure password field appears
when needed. Clusters that require a login code prompt for it separately.
The authenticated session is reused by later actions and background workers,
including on Windows. Codes are not saved. If the session expires, return to
**Connections → Cluster**, sign in again, then reconnect the affected task.

After sign-in, review **Work folder**, **Scheduler** and the detected software.
When detection is incomplete, use **Connections → Software** to enter the
site's launcher, partition or queue, and resources. Ask your cluster support
for a working job example if these are unknown.

VASP needs your licensed POTCAR folder. If QE is missing, use **Install Quantum
ESPRESSO**: check the partition, account and module commands, click **Plan the
build**, review it, then confirm **Start the build**. The agent can prepare
this plan; the app's confirmation submits the build job. Compatible compilers,
MPI and access to source downloads are required. Check progress with **Check
the build**. [Installation details](install.md#your-first-cluster-setup).

## 4. Describe, review and submit

1. Return with **Back to chat**. Open **Files and runs → Files → Add files**, select
   a CIF or POSCAR, and click **Import files**.
2. Describe the calculation, including the engine if you have a preference.
   Answer questions and request changes in the same message box.
3. Open **Tasks** to select a workflow and step. Check its plan and environment.
   The engine belongs to each step, so one project can combine supported codes.
4. Click **Prepare inputs**. Review the generated inputs and resources.
5. Tick **I reviewed the inputs and resources.** and click **Submit**.

Planning and preparing inputs do not submit a calculation. Each submitted
attempt keeps its inputs and history; revising a task does not overwrite an
older run.

## 5. Follow results and continue

Use **Tasks → Refresh status** and **Files and runs → Jobs** to check progress and
logs. **Files and runs → Results** opens the selected run's plots, tables and retained
outputs. Use **Prepare download** to create a results archive where offered.
POTCAR is excluded from shared archives; large restart files remain on the cluster.

Select runs under **Attach results** and ask about them in chat. You can request
a follow-up calculation from a completed stage; review and submit its new
inputs before it runs. After an interruption, sign in to the cluster again if
needed. **Job controls → Reconnect** checks the existing jobs; **Resume monitoring**
restarts the worker and continues the same run's remaining approved stages.
**Cancel this calculation → Cancel job** requests cancellation of the selected attempt.

Closing the browser does not cancel submitted jobs. Keep the computer running
and connected for local monitoring and subsequent stages. Project files and
history are kept under `~/.dft-agent/`; see [storage and updates](install.md#files-and-updates).

## Classic workbench

Open **Settings → Classic workbench** for the previous interface, manual
preparation, sample runs and the Memory browser. Its **New calculation**,
**Cluster setup** and **Runs** instructions belong to that view.

For an offline tour, use **Runs → Load sample runs** there. Result explanations
still use your model account. [Workflows](workflows.md), [QE](qe.md) and
[CLI examples](examples.md) cover the solver-specific controls and commands.
