# Project workspace

This records the project workspace in 0.4.4. The dated checks below distinguish
automated tests, real calculations and installation checks.

## Existing architecture

The Streamlit interface and CLI share Python planners and calculation executors.
VASP and QE have separate input builders, scientific checks, result readers and
continuation paths. `workflow.py`, `qe_workflow.py`, `dispatch.py` and `runtime.py`
freeze inputs, record submission receipts, reconcile uncertain submissions and
check collected outputs. These remain the execution authority.

`agent.py` handles model protocols; the task and QE planners translate goals into
checked plans. Guidance and experience retrieval supply bounded context.
`explanation.py` and the QE context reader ground answers in measured output.
The new task conversation layer keeps revisions, explanations and continuation
requests together. Its immutable engine and source describe an individual task,
not a research project.

The original capabilities preserved here include VASP and QE calculations, structure
edits, batches, magnetic comparisons, restart and repair, downloadable raw
outputs, plots, optional postprocessing, QE installation, and the bounded
electron–phonon, EPW and model-pairing workflows. Their scientific validation
rules remain in force. A successful process is not evidence of convergence or
of superconductivity.

## Gaps at the start of development in the audit

- The UI uses a global run folder and cluster configuration. It has no project
  identity, isolated project memory or workflow connecting different engines.
- Provider keys are stored by provider name. Multiple accounts, connection
  revocation and project associations need explicit identifiers.
- Before this work, Claude Code CLI and keyless local models needed adapters;
  these now have connection paths and automated tests, with live limits below.
- Unix SSH reuses keys, agents and authenticated MFA control sockets. Guided
  sign-in, Windows jump hosts and a Windows MFA session service are now included.
  Jump hosts use their own keys; target passwords are never sent to them.
- The original execution path assumed Slurm. PBS now has a scheduler adapter
  and automated tests; live acceptance still uses Slurm.
- VASP and QE continuation exists within each engine. Cross-engine transfer must
  validate geometry and preserve site order and coordinate frames; magnetic
  parameters, pseudopotentials and restart files are not interchangeable.
- Wannier/EPW are existing QE subworkflows. A general Wannier or Python step must
  have its own reviewed inputs, execution record and output checks.

## Decisions

1. Add a private SQLite workspace index above existing run directories. Use
   versioned transactional migrations, foreign keys and revision checks. Keep
   frozen plans, configuration hashes, execution receipts and parsed artifacts
   authoritative; a database status cannot certify scientific success.
2. Give each project a UUID and separate files, runs, memory, configuration and
   artifact directories. Every session, workflow, step and run is project-bound.
   Resolve paths through the owning project and reject symlink escapes.
3. Store project sessions separately from individual task conversations. One
   project session can coordinate multiple engines and task conversations.
4. Represent workflows as acyclic dependencies between typed steps. Transfer
   only checked artifacts. VASP-to-QE geometry transfer requires a receipt and
   fresh target-engine inputs; never reuse WAVECAR, CHGCAR or QE restart data
   across engines.
5. Keep reusable connections separate from project configuration. Resolve a
   frozen execution profile and a project-specific remote directory for each
   new run. Authentication, connectivity and job state are separate facts.
6. Store credentials in the system keychain under connection IDs, never in
   project JSON, chat or model context. A revoked connection must fail before
   credential loading or environment-variable fallback.
7. Preserve the existing scientific planners, validators, parsers and execution
   state machines behind explicit project tools. Submission and cancellation
   require authorization; reconnect only reconciles existing jobs.
8. Build the project interface around the same backends: navigation on the left,
   conversation and actions in the centre, a hideable context panel on the right.
   Keep advanced forms available where conversational setup cannot infer a
   necessary setting.

## Use the workspace

Create a project. The empty conversation shows three steps.

1. **Model.** Open **Connections → Model**. Installed Codex CLI and Claude Code
   CLI copies are listed with their version and sign-in state; a signed-in CLI
   connects with one click and uses that subscription. When several copies are
   installed, the newest working one is used and remembered with the
   connection. Alternatively choose a provider, keep or edit the suggested
   model name, enter the API key and click **Connect**. The key goes to the
   system keychain; the test result shows the provider's actual answer.
2. **Cluster.** Open **Connections → Cluster** and paste the SSH address
   (`ssh name@host`, `name@host:port` or a bare host). The app first tries your
   SSH keys and agent. If the cluster asks for a password, a password field
   appears. If the cluster then asks for a login code, the app shows the
   cluster's own prompt and a code field; the signed-in session stays open in a
   private multiplexing socket for about eight hours of activity, so later
   commands and detached job monitors reuse it without another code. After the
   connection test the work folder defaults to `~/dft-agent` and the scheduler
   is detected. The app then reads partitions, modules, VASP executables, POTCAR
   folders and `pw.x` from the login node and creates default VASP and QE
   environments; it asks only for what it could not find, for example the
   POTCAR folder. **Sign in through a terminal instead** remains available for
   unusual login flows, and **Software** still allows manual environments.
3. **Task.** Import a structure in **Files** (or let the model use a project
   file) and describe the calculation. A workflow is created and its first step
   is planned immediately; the step card under the conversation shows the plan,
   open questions, **Prepare inputs**, the input review and **Submit**. When a
   project has exactly one matching software environment it is used without
   asking. Planning works before a cluster is connected; preparing inputs needs
   an environment.

A project can combine VASP, QE, ABINIT, CP2K and Elk calculations with Wannier90 and Python steps. **Submit** becomes
available only after reviewing the prepared inputs and resources. **Refresh
status** reads the current state. Under **Job controls**, **Reconnect** checks
existing jobs; **Resume monitoring** restarts the worker and continues the same
run's remaining approved stages. Sign in to the cluster again first if needed.
**Settings** switches the interface language.

## Automatic mode

With a Codex CLI or Claude Code connection the chat runs in **Automatic** mode by
default; the mode switch in the project header returns to **Step by step**. In
automatic mode each message is one CLI run (`codex exec` or `claude -p`) whose only tools
are the project's own, served by `dft-agent mcp` over stdio: project overview,
reading and drafting files, describing structures, creating workflows, running
the engine planners, preparing and reviewing inputs, refreshing status,
explaining verified results, cluster status, software detection, QE
installation and pseudopotentials. The CLI's built-in shell and browsing tools stay
disabled (`--sandbox read-only` and `features.shell_tool=false` for Codex;
`--tools ""` with `--allowedTools mcp__dft__*` for Claude Code). Codex needs
`mcp_servers.dft.default_tools_approval_mode = "approve"`, which the app sets;
without it every tool call is rejected under `approval_policy = "never"`. The
conversation history is passed with each turn; sessions are not persisted by
the CLIs. Tool calls are listed under each reply. There is no submit tool: the
agent prepares and reviews, then asks for the click. API providers always use
the guided mode because they have no agent loop of their own.

## Quantum ESPRESSO installation

When a connected cluster has neither VASP nor QE, the cluster page offers
**Install Quantum ESPRESSO**. **Plan the build** reads the login node's PATH,
then tries module combinations named by `module avail` (compiler and MPI
families) in fresh shells until an MPI Fortran compiler and make are found;
the resulting `module load` line is kept for the build and for running QE. The
plan shows version, prefix, partition, account, CPUs and time limit. **Start
the build** stages the pinned sources and submits one scheduler job; **Check
the build** follows it and, when the manifest reports `installed`, creates the
QE environment with the new `bin` folder and a UPF folder under the work
folder. The agent can inspect the plan and status (`qe_install_plan`,
`qe_install_status`); starting the build requires the button in Connections. A QE step whose
elements lack pseudopotentials offers **Install public UPFs** (SSSP or
PseudoDojo) on the task card; the agent can call `install_pseudopotentials`.
Toolchain detection passed a read-only Paracloud probe. Module fallback and
build failure paths have automated tests; the managed build itself was accepted
on Paracloud in 0.4.3.

Conversations, workflows, files and notes are saved under their project. The
context panel contains files, the workflow, jobs and results; it can be hidden
or resized. Generated run inputs are read-only. Draft edits require the original
file revision, and model-proposed edits need **Apply change**. **Classic workbench**
opens the previous interface for existing path-based runs. Archive and delete
retain the project files and provide a restore action.

## Migration

An upgrade creates the workspace database without moving or rewriting existing
run folders. Existing CLI commands and saved runs remain available. Imports must
be explicit, select a destination project, verify file containment and preserve
provenance. A running or uncertain legacy job must not be copied into a second
submit-capable record. Imported memory must remain owned by its destination
project; public bundled guidance can be shared.

Project deletion is a reversible tombstone, not recursive file removal. Active
or uncertain jobs must be reconciled before deleting their project. Existing
release tags, packages and assets remain unchanged.

## Acceptance on 14 October 2026

On this date the workspace was a local preview; no release was published.

The final local regression suite passed **2,209 tests** in 137 seconds. The
suite includes simulated provider and scheduler responses; the real calls and
cluster jobs are listed separately below.

| Area | Evidence | Scope |
| --- | --- | --- |
| Projects | SQLite migrations, separate paths, reopened sessions, three-project isolation, archive/restore and stale-edit tests | Local storage and access checks |
| Connections | Real Codex request, fresh password SSH connections and macOS keychain save/revoke/delete | Codex and this Slurm account |
| VASP to QE | A real Si VASP relaxation followed by a fresh QE SCF using its accepted geometry | Both jobs completed and passed their output checks |
| Conversation | Actual Codex workflow planning, parameter revisions, single-run and two-run result answers | Answers cite saved evidence; cross-engine energy ranking is disallowed |
| Wannier90 | Pinned official Pb example; failed attempt retained, corrected wrapper retried successfully | Localization and Hamiltonian consistency, not material accuracy |
| Python steps | A real producer calculates Si cell volume; its JSON becomes the next step's input for a CSV | Execution and exact data transfer, not scientific validation |
| Interface | Browser inspection and widget tests for projects, setup, files, history and evidence display | Local macOS browser |
| Packaging | Isolated wheel installation, CLI commands, migrations and packaged UI restart | Python package; desktop executables have not been rebuilt |

The live VASP/QE test found two planning issues: workflow instructions were
mistaken for calculation stages, and an accepted structure was requested again.
Both were corrected and the workflow was repeated through input review. The
Wannier test also exposed an old compute-node Python incompatibility; the fix
was checked on that Python version before retrying. Original attempts remain
available.

The VASP relaxation used ENCUT 520 eV, a 9 × 9 × 9 mesh and a 0.03 eV/angstrom
force threshold. Its final energy was −5.4216444 eV/atom. QE used 60/480 Ry
cutoffs, the transferred cell and sites, and a fresh electronic state. These
runs test the workflow and convergence checks at the chosen settings; they are
not a cutoff or k-point convergence study. Energies from different engines and
pseudopotentials must not be ranked directly.

The Wannier example is pinned to the official repository commit
`1d6b187374a2d50b509e5e79e2cab01a79ff7ce1`.
Its four-orbital Hamiltonian passed the localization and Hermiticity checks.
Python outputs remain labelled `completed_unverified`, even when downstream
steps can consume their verified files.

## Validation limits

- Claude Code, other API providers and local models have adapter tests, but
  were not authenticated and exercised live in this project acceptance run.
- PBS submission and production-cluster MFA still need live site acceptance.
  Windows transport, including a local MFA server, is covered by the
  [0.4.4 release checks](validation-0.4.4.md).
- Desktop installation checks are recorded separately in [0.4.4 validation](validation-0.4.4.md).
- Broader materials, long workflows and subjective interface quality still
  need user review. The successful examples do not certify arbitrary tasks.

Password and login-code sign-in run inside the app on macOS and Linux through
OpenSSH's askpass interface: the password is given only to a password prompt,
the code only to a code prompt, and any other prompt is shown to the user
instead of being answered. The authenticated master connection persists in the
connection's private socket; **Revoke** closes it. Interactive terminal sign-in
remains available and uses the same socket. Windows authenticates through
Paramiko and a local authenticated session service. After one password/key
and verification-code login, background workers reuse that connection. Codes
are not saved or replayed. Disconnection or expiry requires a fresh sign-in. If a terminal is unavailable, setup shows a manual command.

The in-app code flow was exercised against a loopback SSH server that asks for
a password and then a verification code (keyboard-interactive), including
wrong-password and wrong-code paths and reuse of the socket by a second
transport without credentials. It has not yet been run against a production
cluster with multi-factor login.

Provider integration follows the official
[Codex CLI reference](https://learn.chatgpt.com/docs/developer-commands?surface=cli)
and [Claude Code CLI reference](https://code.claude.com/docs/en/cli-reference),
checked against installed CLI help. Authentication availability and a successful
model request are recorded separately.
