from __future__ import annotations

import argparse
from contextlib import ExitStack
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
import threading
import webbrowser

from . import __version__
from .config import ClusterConfig
from .paths import default_config_path, default_runs_root, migrate_legacy_config
from .providers import PROVIDERS

DEFAULT_CONFIG = default_config_path()
CHECK_LABELS = {
    "python3": "Python 3 on the login node",
    "sbatch": "Slurm: sbatch",
    "squeue": "Slurm: squeue",
    "sacct": "Slurm: sacct",
    "scancel": "Slurm: scancel",
    "qsub": "PBS: qsub",
    "qstat": "PBS: qstat",
    "qdel": "PBS: qdel",
    "potcar_root": "POTCAR folder",
    "remote_parent": "Parent of the remote run folder",
    "vasp": "VASP command",
    "vasp_ncl": "SOC / noncollinear command",
    "gpu_partition": "GPU partition",
}
FIX_HINTS = {
    "python3": "Load a Python 3 module in the environment setup commands.",
    "sbatch": "Slurm commands are missing from the PATH. Use a login node, or add a module to the setup commands.",
    "squeue": "Slurm commands are missing from the PATH. Use a login node, or add a module to the setup commands.",
    "sacct": "Slurm accounting is unavailable. Ask your cluster support; monitoring needs sacct.",
    "scancel": "Slurm commands are missing from the PATH. Use a login node, or add a module to the setup commands.",
    "potcar_root": "Check the POTCAR path and your VASP licence, or run Detect from cluster.",
    "remote_parent": "Create the parent folder on the cluster, or choose another remote run folder.",
    "vasp": "Add the VASP module to the setup commands, or use the full path to vasp_std in the VASP command.",
    "vasp_ncl": "Check the noncollinear command, or leave it blank when SOC is not needed.",
    "gpu_partition": "Choose an available GPU partition and a supported number of GPUs per node.",
}


def doctor(config, transport=None):
    if config.engine == "qe":
        from .qe_setup import doctor as qe_doctor
        return qe_doctor(config, transport=transport)
    from .transport import SSHTransport
    owned = transport is None
    remote = transport or SSHTransport(config)
    checks = {}
    try:
        from .scheduler import get_scheduler
        for executable in ("python3", *get_scheduler(config).required_commands):
            result = remote.run("command -v " + executable)
            checks[executable] = {"ok": result.returncode == 0, "detail": result.stdout.strip() or result.stderr.strip()}
        for name, path in (("potcar_root", config.potcar_root), ("remote_parent", str(Path(config.remote_root).parent))):
            result = remote.run("test -d " + shlex.quote(path))
            checks[name] = {"ok": result.returncode == 0, "detail": path}
        if config.compute_backend == "gpu":
            result = remote.run("sinfo -h -p " + shlex.quote(config.partition) + " -o '%a|%G'")
            available = False
            for line in result.stdout.splitlines():
                state, _, resources = line.partition("|")
                counts = re.findall(r"(?:^|,)gpu(?::[^,:()]+)?:(\d+)", resources)
                if state.strip() == "up" and any(int(count) >= config.gpus_per_node for count in counts):
                    available = True
            checks["gpu_partition"] = {
                "ok": result.returncode == 0 and available,
                "detail": result.stdout.strip() or result.stderr.strip() or "No GPU resources were listed.",
            }
        # Launch commands come from the user's configuration.
        words = shlex.split(config.vasp_command)
        solver = next((w for w in words if "vasp" in Path(w).name.lower()), "")
        if solver:
            command = "\n".join(["set -e", *config.setup_commands, "command -v " + shlex.quote(solver)])
            result = remote.run(command)
            checks["vasp"] = {"ok": result.returncode == 0, "detail": result.stdout.strip() or result.stderr.strip()}
        else:
            checks["vasp"] = {"ok": False, "detail": "VASP not found in launch command. Check any wrapper script."}
        if config.vasp_ncl_command:
            words = shlex.split(config.vasp_ncl_command)
            solver = next((w for w in words if "vasp" in Path(w).name.lower()), "")
            if solver:
                result = remote.run("\n".join(["set -e", *config.setup_commands, "command -v " + shlex.quote(solver)]))
                checks["vasp_ncl"] = {"ok": result.returncode == 0, "detail": result.stdout.strip() or result.stderr.strip()}
            else:
                checks["vasp_ncl"] = {"ok": False, "detail": "Check the noncollinear launch command."}
        return {"ok": all(value["ok"] for value in checks.values()), "checks": checks}
    finally:
        if owned:
            remote.close()


def doctor_rows(report):
    """One row per check with a plain label, a pass mark and a fix hint."""
    rows = []
    for name, check in (report.get("checks") or {}).items():
        ok = bool(check.get("ok"))
        rows.append({"check": CHECK_LABELS.get(name, name), "ok": ok, "detail": str(check.get("detail") or "")[:300],
                     "hint": "" if ok else FIX_HINTS.get(name, "Review the detail and your cluster settings.")})
    return rows


def _describe(exc):
    """A readable error line for the terminal."""
    if isinstance(exc, OSError) and getattr(exc, "filename", None) and exc.strerror and not exc.args[1:2] == (None,):
        if isinstance(exc, FileNotFoundError):
            return f"File not found: {exc.filename}"
        if isinstance(exc, PermissionError):
            return f"Permission denied: {exc.filename}"
        return f"{exc.strerror}: {exc.filename}"
    return str(exc) or type(exc).__name__


def _load_optional_config(explicit, default):
    """The cluster settings when they exist. Missing default settings leave the run unattached."""
    if explicit is not None:
        return ClusterConfig.load(explicit)
    if Path(default).expanduser().is_file():
        return ClusterConfig.load(default)
    print(f"No cluster settings at {default}; inputs are prepared for review only. "
          "Attach a cluster later with: dft-agent attach RUN_DIR --config cluster.json", file=sys.stderr)
    return None


def _open_browser_later(url, delay=1.5):
    timer = threading.Timer(delay, webbrowser.open, [url])
    timer.daemon = True
    timer.start()
    return timer


def main():
    default_config = default_config_path()
    parser = argparse.ArgumentParser(description="Run DFT calculations on Slurm.")
    parser.add_argument("--version", action="version", version=f"dft-agent {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    from .project_cli import add_parser as add_project_parser
    add_project_parser(commands)
    commands.add_parser("mcp", help="Serve the current project's tools to Codex CLI or Claude Code over stdio (set DFT_AGENT_PROJECT)")
    broker = commands.add_parser("ssh-broker", help="Manage the local SSH session service")
    broker.add_argument("arguments", nargs=argparse.REMAINDER)
    ui = commands.add_parser("ui", help="Open app")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--no-browser", action="store_true", help="Do not open a browser window")
    init = commands.add_parser("init", help="Set up cluster")
    init.add_argument("--config", type=Path, default=default_config)
    probe = commands.add_parser("doctor", help="Check cluster and VASP")
    probe.add_argument("--config", type=Path, default=default_config)
    probe.add_argument("--password", action="store_true", help="Ask for your SSH password")
    probe.add_argument("--table", action="store_true", help="Print a checklist instead of JSON")
    detect = commands.add_parser("detect", help="Find partitions, accounts, VASP and POTCAR on a cluster")
    detect.add_argument("--host", help="SSH host; defaults to the saved cluster settings")
    detect.add_argument("--user", help="SSH user")
    detect.add_argument("--port", type=int, default=None)
    detect.add_argument("--config", type=Path, default=default_config, help="Saved settings for host, user and port")
    detect.add_argument("--setup", nargs="*", default=None, help="Environment commands to run first")
    detect.add_argument("--password", action="store_true", help="Ask for your SSH password")
    plan = commands.add_parser("plan", help="Plan from a goal")
    plan.add_argument("structure", type=Path)
    plan.add_argument("goal")
    plan.add_argument("--previous", type=Path, help="Revise a saved calculation plan")
    plan.add_argument("--provider", choices=tuple(PROVIDERS), default="responses")
    plan.add_argument("--model", default="")
    plan.add_argument("--base-url", default="")
    plan.add_argument("--output", type=Path, default=Path("plan.json"))
    plan.add_argument("--runs", type=Path, help="Past runs for method guidance")
    plan.add_argument("--engine", choices=("vasp", "qe"), default="vasp")
    plan.add_argument("--config", type=Path, default=default_config)
    plan.add_argument("--images", type=Path, nargs="+", help="QE NEB endpoints or images")
    task = commands.add_parser("task", help="Plan edits, stages and comparisons")
    task.add_argument("structure", type=Path)
    task.add_argument("goal")
    task.add_argument("--previous", type=Path, help="Revise a saved task")
    task.add_argument("--provider", choices=tuple(PROVIDERS), default="responses")
    task.add_argument("--model", default="")
    task.add_argument("--base-url", default="")
    task.add_argument("--output", type=Path, default=Path("task.json"))
    task.add_argument("--runs", type=Path)
    task.add_argument("--engine", choices=("vasp", "qe"), default="vasp")
    task.add_argument("--config", type=Path, default=default_config)
    task.add_argument("--images", type=Path, nargs="+", help="QE NEB endpoints or images")
    for planner in (plan, task):
        planner.add_argument("--pairing", action="store_true", help="Plan a supplied single-band Hamiltonian; requires --engine qe")
        planner.add_argument("--origin", choices=("model", "wannier_unvalidated"), help="Hamiltonian provenance for --pairing")
    qe_command = commands.add_parser("qe", help="Install and check Quantum ESPRESSO")
    qe_command.add_argument("action", choices=("inspect", "install-plan", "install", "install-status", "pseudos"))
    qe_command.add_argument("--config", type=Path, default=default_config)
    qe_command.add_argument("--plan", type=Path, default=Path("qe-install-plan.json"))
    qe_command.add_argument("--receipt", type=Path, default=Path("qe-install-receipt.json"))
    qe_command.add_argument("--prefix")
    qe_command.add_argument("--build-cpus", type=int, default=8)
    qe_command.add_argument("--include-epw", action="store_true", help="Build EPW and Wannier90")
    qe_command.add_argument("--library", choices=("sssp", "pseudodojo"), default="sssp", help="Public UPF library")
    qe_command.add_argument("--elements", nargs="+")
    qe_command.add_argument("--password", action="store_true")
    structure_plan = commands.add_parser("structure-plan", help="Plan structure edits")
    structure_plan.add_argument("structure", type=Path)
    structure_plan.add_argument("goal")
    structure_plan.add_argument("--previous", type=Path, help="Revise a saved structure plan")
    structure_plan.add_argument("--provider", choices=tuple(PROVIDERS), default="responses")
    structure_plan.add_argument("--model", default="")
    structure_plan.add_argument("--base-url", default="")
    structure_plan.add_argument("--output", type=Path, default=Path("structure-plan.json"))
    memory = commands.add_parser("memory", help="Browse or import memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    memory_list = memory_commands.add_parser("list", help="Search records")
    memory_list.add_argument("--query", default="")
    memory_list.add_argument("--status", choices=("verified", "candidate", "shadow_verified", "rejected", "deprecated"))
    memory_list.add_argument("--runs", type=Path)
    memory_show = memory_commands.add_parser("show", help="Read a record")
    memory_show.add_argument("id")
    memory_show.add_argument("--runs", type=Path)
    memory_import = memory_commands.add_parser("import", help="Preview or import selected local records")
    memory_import.add_argument("source", type=Path)
    memory_import.add_argument("--ids", nargs="+", help="Import only these IDs; omit to preview")
    memory_import.add_argument("--runs", type=Path, required=True)
    samples = commands.add_parser("samples", help="Copy completed example runs into the run folder")
    samples.add_argument("--runs", type=Path, default=None, help="Run folder; defaults to the app's run folder")
    samples.add_argument("--list", action="store_true", help="Describe the bundled samples without copying")
    key = commands.add_parser("key", help="Remember or forget an API key in the system keychain")
    key_commands = key.add_subparsers(dest="key_command", required=True)
    key_set = key_commands.add_parser("set", help="Store a key; it is read from the terminal, not from arguments")
    key_set.add_argument("provider", choices=tuple(name for name in PROVIDERS if name != "codex"))
    key_clear = key_commands.add_parser("clear", help="Remove a stored key")
    key_clear.add_argument("provider", choices=tuple(name for name in PROVIDERS if name != "codex"))
    key_commands.add_parser("status", help="Show which providers have a stored key")
    structure_prep = commands.add_parser("prepare-structure", help="Apply reviewed structure edits")
    structure_prep.add_argument("structure", type=Path)
    structure_prep.add_argument("output_dir", type=Path)
    structure_prep.add_argument("--plan", type=Path, required=True)
    convert = commands.add_parser("convert", help="Convert CIF or POSCAR")
    convert.add_argument("structure", type=Path)
    convert.add_argument("output_dir", type=Path)
    convert.add_argument("--format", choices=("cif", "poscar", "both"), default="both")
    explain = commands.add_parser("explain", help="Explain saved results")
    explain.add_argument("run_dir", type=Path)
    explain.add_argument("--question", default="Explain the results.")
    explain.add_argument("--provider", choices=tuple(PROVIDERS), default="responses")
    explain.add_argument("--model", default="")
    explain.add_argument("--base-url", default="")
    next_step = commands.add_parser("next", help="Plan the next QE step from saved results")
    next_step.add_argument("run_dir", type=Path)
    next_step.add_argument("question")
    next_step.add_argument("--provider", choices=tuple(PROVIDERS), default="responses")
    next_step.add_argument("--model", default="")
    next_step.add_argument("--base-url", default="")
    next_step.add_argument("--output", type=Path, default=Path("next-step.json"))
    prepare_next = commands.add_parser("prepare-next", help="Prepare a reviewed QE continuation or repair")
    prepare_next.add_argument("run_dir", type=Path)
    prepare_next.add_argument("new_run_dir", type=Path)
    prepare_next.add_argument("--plan", type=Path, required=True)
    repair = commands.add_parser("repair", help="Plan a bounded repair")
    repair.add_argument("run_dir", type=Path)
    repair.add_argument("--provider", choices=tuple(PROVIDERS), default="responses")
    repair.add_argument("--model", default="")
    repair.add_argument("--base-url", default="")
    repair.add_argument("--output", type=Path, default=Path("repair.json"))
    repair_prep = commands.add_parser("prepare-repair", help="Prepare reviewed repair inputs")
    repair_prep.add_argument("run_dir", type=Path)
    repair_prep.add_argument("new_run_dir", type=Path)
    repair_prep.add_argument("--plan", type=Path, required=True)
    prep = commands.add_parser("prepare", help="Prepare inputs locally; a cluster is optional until submission")
    prep.add_argument("structure", type=Path)
    prep.add_argument("run_dir", type=Path)
    prep.add_argument("--config", type=Path, default=None, help=f"Cluster settings (default: {default_config} when present)")
    source = prep.add_mutually_exclusive_group()
    source.add_argument("--task", choices=("relax", "vc-relax", "scf", "nscf", "bands", "dos", "pdos", "phonon", "phonon_bands", "electron_phonon", "wannier_electron_phonon", "neb", "pp"))
    source.add_argument("--plan", type=Path)
    prep.add_argument("--magnetic-states", nargs="+", choices=("NM", "FM", "AFM"))
    prep.add_argument("--parameters", default="{}", help="Calculation settings as a JSON object")
    prep.add_argument("--images", type=Path, nargs="+", help="QE NEB endpoints or images")
    prep.add_argument("--cluster-python", help="Existing cluster Python executable for a reviewed pairing plan")
    attach = commands.add_parser("attach", help="Attach cluster settings to a prepared run or batch")
    attach.add_argument("run_dir", type=Path)
    attach.add_argument("--config", type=Path, default=default_config)
    continuation = commands.add_parser("continue", help="Prepare from an accepted stage")
    continuation.add_argument("source_run", type=Path)
    continuation.add_argument("stage", help="Stage folder, for example 01_relax")
    continuation.add_argument("run_dir", type=Path)
    continuation.add_argument("--tasks", nargs="+", choices=("relax", "vc-relax", "scf", "nscf", "bands", "dos", "pdos", "phonon", "phonon_bands", "electron_phonon", "epw_iso", "epw_tc", "epw_aniso", "pp"), required=True)
    continuation.add_argument("--parameters", default="{}", help="Changes to inherited settings (JSON)")
    continuation.add_argument("--stage-parameters", default="null", help="Per-stage changes (JSON list)")
    extra = commands.add_parser("files", help="List or collect additional cluster outputs")
    extra.add_argument("run_dir", type=Path)
    extra.add_argument("stage", help="Stage folder, for example 03_bands")
    extra.add_argument("--fetch", nargs="+", help="Output filenames to collect")
    extra.add_argument("--password", action="store_true")
    post = commands.add_parser("postprocess", help="Export bands and DOS with cluster VASPKIT")
    post.add_argument("run_dir", type=Path)
    post.add_argument("stage", help="Stage folder, for example 03_bands")
    post.add_argument("--tasks", nargs="+", choices=("bands", "projected_bands", "total_dos", "projected_dos"))
    post.add_argument("--executable", help="VASPKIT executable on the cluster")
    post.add_argument("--password", action="store_true")
    command_help = {
        "watch": "Submit and monitor",
        "watch-batch": "Submit and monitor a batch",
        "resume": "Reconnect and monitor",
        "status": "Show saved status",
        "cancel": "Cancel job",
        "bundle": "Export results ZIP",
    }
    for name, help_text in command_help.items():
        sub = commands.add_parser(name, help=help_text)
        sub.add_argument("run_dir", type=Path)
        if name in {"watch", "watch-batch", "resume", "cancel"}:
            sub.add_argument("--password", action="store_true", help="Ask for your SSH password")
        if name in {"watch", "watch-batch"}:
            sub.add_argument("--interval", type=float, default=20)
    args = parser.parse_args()
    try:
        migrate_legacy_config()
    except OSError:
        pass
    credential_scope = ExitStack()
    try:
        if args.command == "project":
            from .project_cli import dispatch as project_dispatch
            return project_dispatch(args)
        if args.command == "mcp":
            from .mcp_server import main as mcp_main
            return mcp_main()
        if args.command == "ssh-broker":
            from .ssh_broker import main as broker_main
            return broker_main(args.arguments)
        if getattr(args, "password", False):
            from .transport import password_context
            credential_scope.enter_context(password_context(getpass.getpass("SSH password: ")))
        if args.command == "ui":
            if not 1024 <= args.port <= 65535:
                parser.error("Choose a local port between 1024 and 65535")
            from streamlit.web import cli as streamlit_cli
            url = f"http://127.0.0.1:{args.port}"
            sys.argv = ["streamlit", "run", str(Path(__file__).with_name("app.py")), "--server.address=127.0.0.1",
                        f"--server.port={args.port}", "--server.headless=true", "--browser.gatherUsageStats=false",
                        "--client.toolbarMode=minimal"]
            print(f"DFT Agent {__version__} · {url} · press Ctrl+C to stop", file=sys.stderr)
            if not args.no_browser and not os.environ.get("DFT_AGENT_NO_BROWSER"):
                _open_browser_later(url)
            return streamlit_cli.main()
        if args.command == "init":
            if args.config.expanduser().exists():
                raise FileExistsError("This configuration file already exists. Edit it or choose a different path with --config.")
            cfg = ClusterConfig(host=input("SSH host: ").strip(), user=input("SSH user: ").strip(), remote_root=input("Remote run folder (absolute path): ").strip(), vasp_command=input("VASP command [srun vasp_std]: ").strip() or "srun vasp_std", potcar_root=input("POTCAR folder (absolute path): ").strip(), partition=input("Slurm partition: ").strip())
            print(cfg.save(args.config))
            return 0
        if args.command == "doctor":
            result = doctor(ClusterConfig.load(args.config))
            if args.table:
                for row in doctor_rows(result):
                    print(f"{'OK  ' if row['ok'] else 'FAIL'} {row['check']}: {row['detail']}" + (f"\n     {row['hint']}" if row["hint"] else ""))
            else:
                print(json.dumps(result, indent=2))
            return 0 if result["ok"] else 1
        if args.command == "files":
            from .workflow import read_state
            state = read_state(args.run_dir)
            if state.get("workflow") == "pairing":
                from .qe_pairing_workflow import load_run_context
                if args.stage not in {stage["folder"] for stage in state["stages"]}:
                    raise ValueError("Choose the existing pairing stage folder.")
                if args.fetch:
                    raise ValueError("Pairing outputs are already collected. Use dft-agent bundle RUN_DIR to download them.")
                context = load_run_context(args.run_dir)
                result = {"status": context["status"], "artifacts": [item for item in context["artifacts"] if item["status"] == "verified"],
                          "download": "Use dft-agent bundle RUN_DIR to export the model and collected results."}
            elif state.get("engine") == "qe":
                from .qe_workflow import checkpoint_files, fetch_raw_files
                matches = [i for i, stage in enumerate(state["stages"]) if stage["folder"] == args.stage]
                if len(matches) != 1:
                    raise ValueError("Choose an existing QE stage folder.")
                result = (fetch_raw_files(args.run_dir, args.fetch, matches[0]) if args.fetch
                          else checkpoint_files(args.run_dir, matches[0]))
            else:
                from .postprocessing import fetch_outputs, list_remote_outputs
                result = (fetch_outputs(args.run_dir, args.stage, args.fetch) if args.fetch
                          else list_remote_outputs(args.run_dir, args.stage))
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "postprocess":
            from .workflow import read_state
            if read_state(args.run_dir).get("engine") == "qe":
                raise ValueError("QE results are exported during collection. Use next to plan further analysis.")
            from .postprocessing import postprocess_run
            result = postprocess_run(args.run_dir, args.stage, args.tasks, executable=args.executable)
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "complete" else 1
        if args.command == "detect":
            from .discovery import probe_cluster, probe_config
            saved = None
            if Path(args.config).expanduser().is_file() and not (args.host and args.user):
                saved = ClusterConfig.load(args.config)
            host = args.host or (saved.host if saved else "")
            user = args.user or (saved.user if saved else "")
            if not host or not user:
                raise ValueError("Give --host and --user, or save cluster settings first.")
            port = args.port or (saved.port if saved else 22)
            setup = args.setup if args.setup is not None else (saved.setup_commands if saved else [])
            probe = probe_config(host, user, port, connect_timeout=saved.connect_timeout if saved else 15,
                                 ssh_control_path=saved.ssh_control_path if saved else "", setup_commands=setup)
            result = probe_cluster(probe)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "samples":
            from .samples import install_samples, list_samples
            if args.list:
                print(json.dumps(list_samples(), indent=2, ensure_ascii=False))
                return 0
            runs = args.runs or default_runs_root()
            installed = install_samples(runs)
            for path in installed:
                print(path)
            if not installed:
                print(f"The sample runs are already in {runs}.", file=sys.stderr)
            return 0
        if args.command == "key":
            from . import credentials
            if args.key_command == "status":
                from .settings import load_settings
                if not load_settings().get("keychain"):
                    print("No API key is remembered on this computer.")
                    return 0
                stored = [name for name in PROVIDERS if name != "codex" and credentials.stored_key(name)]
                print("Remembered keys: " + (", ".join(PROVIDERS[name]["label"] for name in stored) if stored else "none"))
                return 0
            if args.key_command == "set":
                value = getpass.getpass(f"{PROVIDERS[args.provider]['label']} API key: ")
                credentials.save_key(args.provider, value)
                print(f"Stored the {PROVIDERS[args.provider]['label']} key in the system keychain.")
                return 0
            credentials.delete_key(args.provider)
            print(f"Removed the {PROVIDERS[args.provider]['label']} key from the system keychain.")
            return 0
        if args.command == "memory":
            from .knowledge import load_catalog, preview_import, import_selected
            if args.memory_command == "import":
                result = (import_selected(args.source.expanduser(), args.ids, args.runs)
                          if args.ids else preview_import(args.source.expanduser()))
            else:
                records = load_catalog(args.runs)
                if args.memory_command == "show":
                    result = next((item for item in records if item["id"] == args.id), None)
                    if result is None:
                        raise ValueError("Record not found.")
                else:
                    query = args.query.lower().strip()
                    result = [{key: item[key] for key in ("id", "title", "kind", "status", "origin", "evidence_valid")}
                              for item in records if (not args.status or item["status"] == args.status)
                              and (not query or query in " ".join([item["title"], item["summary"],
                                   *item.get("topics", []), *item.get("formulas", []), *item.get("methods", [])]).lower())]
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command in {"plan", "task"}:
            from .agent import ModelSettings, draft_plan
            from .task_agent import draft_task
            goal, history = args.goal, []
            if args.pairing and (args.engine != "qe" or not args.origin or args.images):
                raise ValueError("Pairing plans require --engine qe and --origin model or wannier_unvalidated, without --images.")
            if args.origin and not args.pairing:
                raise ValueError("Use --origin only with --pairing.")
            if args.previous:
                previous = json.loads(args.previous.read_text())
                if bool(previous.get("kind") == "qe_pairing") != args.pairing:
                    raise ValueError("Keep --pairing when revising a pairing plan; start a new plan to change routes.")
                if args.pairing and previous.get("origin") != args.origin:
                    raise ValueError("The Hamiltonian provenance changed. Start a new pairing plan.")
                if previous.get("source_sha256") != hashlib.sha256(args.structure.read_bytes()).hexdigest():
                    raise ValueError("The source changed. Start a new calculation plan.")
                if not isinstance(previous.get("goal"), str) or not isinstance(previous.get("dialogue"), list):
                    raise ValueError("This plan has no conversation. Start a new calculation plan.")
                goal = previous["goal"]
                history = previous["dialogue"] + [{"role": "user", "content": args.goal}]
            settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
            if args.pairing:
                from .qe_pairing_agent import draft_task as draft_pairing_task
                result = draft_pairing_task(goal, args.structure, settings, origin=args.origin, history=history)
            elif args.engine == "qe":
                from .qe_agent import draft_task as draft_qe_task
                result = draft_qe_task(goal, args.structure, settings, history=history,
                                       config=ClusterConfig.load(args.config), images=args.images, runs_root=args.runs)
            else:
                planner = draft_task if args.command == "task" else draft_plan
                result = planner(goal, args.structure, settings, history=history, runs_root=args.runs)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "ready" else 1
        if args.command == "qe":
            from . import qe_setup
            from dataclasses import replace
            config = ClusterConfig.load(args.config)
            if config.engine != "qe":
                raise ValueError("Select Quantum ESPRESSO in cluster settings.")
            if args.action == "inspect":
                result = qe_setup.inspect(config)
            elif args.action == "install-plan":
                result = qe_setup.install_plan(config, prefix=args.prefix, build_jobs=args.build_cpus, include_epw=args.include_epw)
                args.plan.write_text(json.dumps(result, indent=2) + "\n")
            elif args.action == "install":
                result = qe_setup.start_install(config, json.loads(args.plan.read_text()))
                args.receipt.write_text(json.dumps(result, indent=2) + "\n")
            elif args.action == "install-status":
                result = qe_setup.poll_install(config, json.loads(args.receipt.read_text()))
            else:
                if not args.elements:
                    raise ValueError("Give --elements, for example Si O.")
                result = qe_setup.ensure_pseudos(config, args.elements, library=args.library)
                mapping = dict(config.qe_pseudopotentials)
                for element, item in result["elements"].items():
                    mapping[element] = {"filename": item["filename"], "sha256": item["sha256"],
                        "ecutwfc": item["cutoff_wfc"], "ecutrho": item["cutoff_rho"],
                        "element": element, "relativistic": item["header"]["relativistic"]}
                replace(config, qe_pseudopotentials=mapping).save(args.config)
            print(json.dumps(result, indent=2))
            return 1 if result.get("status") in {"failed", "needs_attention"} else 0
        if args.command == "structure-plan":
            from .agent import ModelSettings
            from .structure_agent import draft_structure
            goal, history = args.goal, []
            if args.previous:
                previous = json.loads(args.previous.read_text())
                if previous.get("source_sha256") != hashlib.sha256(args.structure.read_bytes()).hexdigest():
                    raise ValueError("The source changed. Start a new structure plan.")
                goal = previous["goal"]
                history = previous["dialogue"] + [{"role": "user", "content": args.goal}]
            settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
            result = draft_structure(goal, args.structure, settings, history=history)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "ready" else 1
        if args.command in {"prepare-structure", "convert"}:
            from .structures import apply_structure_plan
            proposal = json.loads(args.plan.read_text()) if args.command == "prepare-structure" else {
                "schema_version": 1, "status": "ready", "operations": [], "output_format": args.format,
                "source_sha256": hashlib.sha256(args.structure.read_bytes()).hexdigest()}
            result = apply_structure_plan(args.structure, args.output_dir, proposal)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command == "explain":
            from .agent import ModelSettings
            from .explanation import explain_run
            settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
            result = explain_run(args.run_dir, settings, question=args.question)
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        if args.command in {"next", "prepare-next"}:
            from . import qe_reasoning
            from .agent import ModelSettings
            if args.command == "next":
                settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
                result = qe_reasoning.draft_next_action(args.question, args.run_dir, settings)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
            else:
                result = qe_reasoning.prepare_next_action(args.run_dir, args.new_run_dir, json.loads(args.plan.read_text()))
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "repair":
            from .agent import ModelSettings
            from .recovery import draft_repair
            settings = ModelSettings(provider=args.provider, model=args.model, base_url=args.base_url or None)
            result = draft_repair(args.run_dir, settings)
            args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if result["status"] == "ready" else 1
        if args.command == "prepare-repair":
            from .recovery import prepare_repair
            result = prepare_repair(args.run_dir, args.new_run_dir, json.loads(args.plan.read_text()))
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0
        from .workflow import attach_config, prepare_run, prepare_plan, read_state, watch, resume, cancel, bundle_run, TERMINAL
        if args.command == "attach":
            config = ClusterConfig.load(args.config)
            if (args.run_dir / "batch.json").is_file():
                from .batch import attach_batch_config
                result = attach_batch_config(args.run_dir, config)
            else:
                result = attach_config(args.run_dir, config)
            print(json.dumps(result, indent=2))
            return 0
        if args.command == "continue":
            if read_state(args.source_run).get("engine") == "qe":
                from .qe_continuation import prepare_continuation
            else:
                from .continuation import prepare_continuation
            result = prepare_continuation(args.source_run, args.stage, args.run_dir,
                                         ClusterConfig.load(args.source_run / "config.json"), args.tasks,
                                         parameters=json.loads(args.parameters), stage_parameters=json.loads(args.stage_parameters))
            print(json.dumps(result, indent=2))
            return 0
        if args.command in {"watch", "watch-batch", "status", "resume", "cancel", "bundle"} and (args.run_dir / "batch.json").is_file():
            from .batch import watch_batch, read_batch, resume_batch, cancel_batch, bundle_batch
            if args.command == "bundle":
                print(bundle_batch(args.run_dir))
                return 0
            if args.command in {"watch", "watch-batch"}:
                if args.interval < 1:
                    raise ValueError("Choose a polling interval of at least one second.")
                result = watch_batch(args.run_dir, args.interval)
            elif args.command == "resume":
                result = resume_batch(args.run_dir)
                if result["status"] not in TERMINAL:
                    result = watch_batch(args.run_dir)
            else:
                result = (cancel_batch if args.command == "cancel" else read_batch)(args.run_dir)
            print(json.dumps(result, indent=2))
            return 1 if result.get("status") in {"failed", "needs_attention"} else 0
        if args.command == "watch-batch":
            raise ValueError("Choose a prepared batch folder.")
        if args.command == "prepare":
            if args.cluster_python and not args.plan:
                raise ValueError("Use --cluster-python with a reviewed pairing --plan.")
            config = _load_optional_config(args.config, default_config)
            if args.plan:
                if args.parameters != "{}" or args.magnetic_states:
                    raise ValueError("Edit the plan before changing its settings.")
                proposal = json.loads(args.plan.read_text())
                if args.cluster_python and proposal.get("kind") != "qe_pairing":
                    raise ValueError("Use --cluster-python only with a pairing plan.")
                if proposal.get("status") != "ready":
                    raise ValueError("Resolve the plan's questions before preparing inputs.")
                if proposal.get("source_sha256") != hashlib.sha256(args.structure.read_bytes()).hexdigest():
                    raise ValueError("The structure differs from the plan. Create a new plan.")
                if proposal.get("kind") == "qe_pairing":
                    from dataclasses import replace
                    from .qe_pairing_agent import prepare_task
                    if not args.cluster_python or config is None or config.engine != "qe":
                        raise ValueError("Pairing preparation requires --cluster-python and a saved QE --config.")
                    if args.images:
                        raise ValueError("A pairing plan uses the supplied Hamiltonian, not --images.")
                    config = replace(config, compute_backend="cpu", nodes=1, tasks=1, cpus_per_task=1)
                    result = prepare_task(args.structure, args.run_dir, config, proposal, python_executable=args.cluster_python)
                elif proposal.get("kind") == "qe_task":
                    from .qe_agent import prepare_task
                    result = prepare_task(args.structure, args.run_dir, config, proposal, images=args.images)
                elif proposal.get("kind") == "task":
                    from .task_agent import prepare_task
                    result = prepare_task(args.structure, args.run_dir, config, proposal)
                else:
                    result = prepare_plan(args.structure, args.run_dir, config, proposal["tasks"], proposal["parameters"],
                                          proposal.get("magnetic_states") or None, proposal.get("initial_moment", 3.0))
                if proposal.get("kind") not in {"task", "qe_task", "qe_pairing"}:
                    (args.run_dir / "proposal.json").write_text(json.dumps(proposal, indent=2, ensure_ascii=False) + "\n")
            elif args.magnetic_states:
                result = prepare_plan(args.structure, args.run_dir, config, [args.task or "scf"],
                                      json.loads(args.parameters), args.magnetic_states)
            else:
                parameters = json.loads(args.parameters)
                if args.images:
                    if config is None or config.engine != "qe" or args.task != "neb":
                        raise ValueError("Use --images with a QE NEB calculation.")
                    if "images" in parameters:
                        raise ValueError("Supply images once, using --images.")
                    parameters["images"] = [str(path.resolve()) for path in args.images]
                result = prepare_run(args.structure, args.run_dir, config, args.task or "relax", parameters)
        elif args.command == "watch":
            if args.interval < 1:
                parser.error("Choose a polling interval of at least one second.")
            result = watch(args.run_dir, args.interval)
        elif args.command == "status":
            result = read_state(args.run_dir)
        elif args.command == "resume":
            result = resume(args.run_dir)
            if result["status"] not in TERMINAL:
                result = watch(args.run_dir)
        elif args.command == "cancel":
            result = cancel(args.run_dir)
        else:
            print(bundle_run(args.run_dir))
            return 0
        print(json.dumps(result, indent=2))
        return 1 if result.get("status") in {"failed", "needs_attention"} else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"Error: {_describe(exc)}", file=sys.stderr)
        return 1
    finally:
        credential_scope.close()


if __name__ == "__main__":
    raise SystemExit(main())
