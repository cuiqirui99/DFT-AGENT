"""Managed ABINIT, CP2K and Elk installations from pinned conda-forge packages.

Nothing is compiled. The cluster's architecture and glibc select a package set
from engine_packages.json; every package is pinned by SHA-256. The login node
downloads them (the app uploads any it cannot reach), a short scheduler job
creates the environment offline with micromamba, checks the bundled data files
and runs a small silicon calculation. Pseudopotentials for ABINIT are installed
per element afterwards, like the Quantum ESPRESSO libraries.
"""
from __future__ import annotations

import contextlib
from functools import lru_cache
import hashlib
from importlib.resources import files
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import ssl
import tempfile
import time
import urllib.request
import uuid

from .scheduler import get_scheduler
from .transport import SSHTransport, TransportError

ENGINES = ("abinit", "cp2k", "elk")
LABELS = {"abinit": "ABINIT", "cp2k": "CP2K", "elk": "Elk"}
_MARKER = "__DFT_ENGINE_SETUP__"
_PLATFORMS = {"x86_64": "linux-64", "amd64": "linux-64", "aarch64": "linux-aarch64", "arm64": "linux-aarch64"}
_SMOKE = {"abinit": {"mesh": [2, 2, 2], "ecut_ha": 12}, "cp2k": {"mesh": [2, 2, 2], "cutoff_ry": 200, "rel_cutoff_ry": 40},
          "elk": {"mesh": [2, 2, 2], "rgkmax": 5.0, "gmaxvr": 10.0}}
_SI = ("data_Si\n_cell_length_a 3.8395898\n_cell_length_b 3.8395898\n_cell_length_c 3.8395898\n"
       "_cell_angle_alpha 60\n_cell_angle_beta 60\n_cell_angle_gamma 60\n_symmetry_space_group_name_H-M 'P 1'\n"
       "loop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n"
       "Si1 Si 0 0 0\nSi2 Si 0.25 0.25 0.25\n")


@lru_cache(maxsize=1)
def catalog():
    return json.loads(files("vasp_slurm_agent").joinpath("engine_packages.json").read_text())


def _absolute(value):
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or any(c in value for c in "\r\n\x00"):
        raise ValueError("Use an absolute cluster path without '..'.")
    return str(path)


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def _connection(config, transport):
    active = transport or SSHTransport(config)
    try:
        yield active
    finally:
        if transport is None:
            active.close()


def _json_run(transport, code, *, setup=(), timeout=60):
    command = "\n".join(["set -e", *setup, "python3 -c " + shlex.quote(code)])
    result = transport.run(command, timeout=timeout)
    if result.returncode:
        raise TransportError((result.stderr or result.stdout).strip()[-2000:] or "Program setup failed.")
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER):])
    raise TransportError("The cluster returned no status. Check that python3 is available on the login node.")


def detect_platform(config, transport=None):
    """Architecture, glibc and the tools an installation needs, from the login node."""
    code = f"""import json,os,platform,shutil
try:
    libc=os.confstr('CS_GNU_LIBC_VERSION') or ''
except (ValueError,OSError):
    libc=''
print({_MARKER!r}+json.dumps({{'machine':platform.machine(),'libc':libc,'system':platform.system(),
    'tools':{{name:bool(shutil.which(name)) for name in ('sbatch','squeue','sacct','qsub','qstat','python3','tar')}}}}))
"""
    with _connection(config, transport) as active:
        info = _json_run(active, code)
    machine = str(info.get("machine", ""))
    libc = str(info.get("libc", ""))
    glibc = libc.split()[-1] if libc.startswith("glibc ") else None
    return {"platform": _PLATFORMS.get(machine.lower()), "machine": machine, "glibc": glibc,
            "system": info.get("system"), "tools": info.get("tools", {})}


def _version_tuple(value):
    return tuple(int(part) for part in str(value).split(".")[:2])


def choose_variant(engine, platform, glibc):
    if engine not in ENGINES:
        raise ValueError("Choose ABINIT, CP2K or Elk.")
    entry = catalog()["engines"][engine]
    if platform not in entry["platforms"]:
        raise ValueError(f"No pinned {LABELS[engine]} package exists for this cluster's architecture ({platform or 'unknown'}). "
                         "Use a module the cluster provides instead.")
    if not glibc:
        raise ValueError("The login node did not report a GNU C library version; a managed install needs Linux with glibc.")
    usable = [variant for key, variant in entry["platforms"][platform].items() if _version_tuple(key) <= _version_tuple(glibc)]
    if not usable:
        raise ValueError(f"{LABELS[engine]} packages need glibc {min(entry['platforms'][platform], key=_version_tuple)} or newer; "
                         f"this cluster has {glibc}. Use a module the cluster provides instead.")
    return max(usable, key=lambda variant: _version_tuple(variant["min_glibc"]))


def library_root(engine, prefix, remote_base):
    if engine == "abinit":
        return str(PurePosixPath(remote_base) / "pseudos" / "abinit")
    if engine == "cp2k":
        # The pinned data files, kept apart from the data folder of whichever CP2K package is installed.
        from .cp2k import catalog as basis
        return str(PurePosixPath(remote_base) / "libraries" / f"cp2k-{basis()['version']}")
    return str(PurePosixPath(prefix) / "share" / "elk" / "species")


def launch_command(engine, prefix, variant):
    program = catalog()["engines"][engine]["program"]
    binary = str(PurePosixPath(prefix) / "bin" / program)
    if engine == "elk":
        return binary
    return f"{shlex.quote(str(PurePosixPath(prefix) / 'bin' / 'mpiexec'))} -n {{tasks}} {shlex.quote(binary)}"


def environment_setup(engine, variant):
    """Shell lines for jobs using a managed environment: its own libraries and launcher."""
    lines = ["unset LD_LIBRARY_PATH"]
    if variant.get("mpi") == "mpich":
        # Start ranks on the allocated node directly; nested srun steps can pin all ranks to one core.
        lines.append("export HYDRA_BOOTSTRAP=fork")
    return lines


def validate_managed_resources(config):
    """Managed MPICH launches ranks on one allocated node, not across nodes."""
    if config is None or config.engine not in {"abinit", "cp2k"} or config.nodes == 1:
        return
    for line in config.setup_commands:
        if shlex.split(line) == ["export", "HYDRA_BOOTSTRAP=fork"]:
            raise ValueError("This managed installation runs on one node. Set Nodes to 1, or use the cluster's MPI build.")


def _smoke_inputs(engine, folder):
    from .engine_workflow import adapter
    folder = Path(folder)
    structure = folder.parent / "Si.cif"
    structure.write_text(_SI)
    metadata = adapter(engine).prepare_inputs(structure, folder, "scf", dict(_SMOKE[engine]))
    structure.unlink()
    return metadata


def install_plan(config, engine, platform_info, *, prefix=None, build_cpus=4, walltime="01:00:00", attempt=0):
    """A reviewed, deterministic plan. Nothing is downloaded or submitted.

    A retry after a failed installation is a new attempt: it gets its own plan ID, folder and
    job, and reuses an environment that an earlier attempt created from the same packages."""
    if engine not in ENGINES:
        raise ValueError("Choose ABINIT, CP2K or Elk.")
    variant = choose_variant(engine, platform_info.get("platform"), platform_info.get("glibc"))
    entry = catalog()["engines"][engine]
    version = variant.get("version", entry["version"])
    remote_base = _absolute(config.remote_root)
    tag = hashlib.sha256("".join(p["sha256"] for p in variant["packages"]).encode()).hexdigest()[:8]
    prefix = _absolute(prefix or str(PurePosixPath(remote_base) / "software" / f"{engine}-{version}-{tag}"))
    if type(build_cpus) is not int or not 1 <= build_cpus <= 64:
        raise ValueError("Use 1 to 64 CPUs for the installation job.")
    if type(attempt) is not int or attempt < 0:
        raise ValueError("The installation attempt must be a whole number.")
    mamba = catalog()["micromamba"][platform_info["platform"]]
    downloads = [{"name": "micromamba-" + catalog()["micromamba"]["version"] + ".tar.bz2",
                  "url": mamba["url"], "sha256": mamba["sha256"], "size": mamba["size"], "role": "micromamba"}]
    downloads += [{"name": p["fn"], "url": p["url"], "sha256": p["sha256"], "size": p["size"], "role": "package"}
                  for p in variant["packages"]]
    extras = {}
    if engine == "elk":
        from .elk import catalog as species
        archive = species()["archive"]
        downloads.append({"name": archive["filename"], "url": archive["url"], "sha256": archive["sha256"],
                          "size": archive["size"], "role": "species_archive"})
        extras["species"] = {name: value["sha256"] for name, value in species()["species"].items()}
        extras["species_dir"] = archive["species_dir"]
    elif engine == "cp2k":
        from .cp2k import catalog as basis
        downloads += [{"name": name, "url": value["url"], "sha256": value["sha256"], "size": value["size"], "role": "library"}
                      for name, value in sorted(basis()["files"].items())]
    else:
        from .abinit import catalog as pseudos
        silicon = pseudos()["libraries"]["pbe_sr"]["elements"]["Si"]
        downloads.append({"name": silicon["filename"], "url": silicon["url"], "sha256": silicon["sha256"],
                          "size": silicon["size"], "role": "library"})
    names = [item["name"] for item in downloads]
    if len(set(names)) != len(names) or any(PurePosixPath(n).name != n or n.startswith(".") for n in names):
        raise ValueError("The pinned package list has unsafe or duplicate names.")
    identity = {"engine": engine, "version": version, "platform": platform_info["platform"], "variant": variant["min_glibc"],
                "prefix": prefix, "partition": config.partition, "account": config.account, "scheduler": config.scheduler,
                "build_cpus": build_cpus, "walltime": walltime, "downloads": [item["sha256"] for item in downloads]}
    if attempt:
        identity["attempt"] = attempt
    plan_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    work = str(PurePosixPath(remote_base) / "software" / ".install" / f"{engine}-{plan_id}")
    return {"schema_version": 1, "id": plan_id, "engine": engine, "version": version, "program": entry["program"],
            "platform": platform_info["platform"], "glibc": platform_info["glibc"], "variant": variant["min_glibc"],
            "mpi": variant["mpi"], "prefix": prefix, "bin_dir": str(PurePosixPath(prefix) / "bin"), "work_dir": work,
            "library_root": library_root(engine, prefix, remote_base), "command": launch_command(engine, prefix, variant),
            "setup_commands": environment_setup(engine, variant), "partition": config.partition, "account": config.account,
            "scheduler": config.scheduler, "extra_pbs": list(config.extra_pbs), "build_cpus": build_cpus, "walltime": walltime,
            "download_size": sum(item["size"] for item in downloads), "downloads": downloads, "extras": extras,
            "attempt": attempt}


_STAGE_PY = r'''import fcntl, hashlib, json, os, pathlib, sys, tempfile, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
root = pathlib.Path(__file__).resolve().parent
# Reconnection may start another launcher while this worker is still downloading.
worker_lock = (root / "stage.lock").open("a")
try:
    fcntl.flock(worker_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(0)
plan = json.loads((root / "plan.json").read_text())
store = root / "pkgs"
store.mkdir(exist_ok=True)
lock = threading.Lock()
state = {"status": "staging", "done": 0, "total": len(plan["downloads"]), "bytes": 0, "failed": [], "started_at": time.time(), "pid": os.getpid()}

def save():
    fd, name = tempfile.mkstemp(prefix=".stage-status-", dir=str(root))
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(state, out)
        os.replace(name, root / "stage-status.json")
    finally:
        if os.path.exists(name): os.unlink(name)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): h.update(chunk)
    return h.hexdigest()

def fetch(item):
    target = store / item["name"]
    if target.is_file() and not target.is_symlink() and target.stat().st_size == item["size"] and sha(target) == item["sha256"]:
        return True
    for attempt in range(3):
        fd, temporary = tempfile.mkstemp(prefix=".download-", dir=str(store))
        os.close(fd)
        try:
            request = urllib.request.Request(item["url"], headers={"User-Agent": "DFT-Agent"})
            with open(temporary, "wb") as out, urllib.request.urlopen(request, timeout=60) as response:
                if not response.geturl().startswith("https://"): raise RuntimeError("HTTPS redirect required")
                size = 0
                while True:
                    data = response.read(1048576)
                    if not data: break
                    size += len(data)
                    if size > item["size"]: raise RuntimeError("Download exceeds the pinned size")
                    out.write(data)
            if sha(pathlib.Path(temporary)) != item["sha256"]: raise RuntimeError("Checksum mismatch")
            os.replace(temporary, target)
            return True
        except Exception:
            time.sleep(2 + 3 * attempt)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
    return False

def run(item):
    ok = fetch(item)
    with lock:
        state["done"] += 1
        if ok: state["bytes"] += item["size"]
        else: state["failed"].append(item["name"])
        save()

save()
with ThreadPoolExecutor(max_workers=4) as pool:
    list(pool.map(run, plan["downloads"]))
state["status"] = "transfer_required" if state["failed"] else "staged"
state["finished_at"] = time.time()
save()
'''

_INSTALL_PY = r'''import fcntl, hashlib, json, os, pathlib, re, shutil, subprocess, sys, tarfile, tempfile, time, uuid
root = pathlib.Path(__file__).resolve().parent
plan = json.loads((root / "plan.json").read_text())
prefix = pathlib.Path(plan["prefix"])
status_path = root / "install-status.json"

def save(value):
    if retained_prefix is not None:
        value["retained_prefix"] = retained_prefix
    fd, name = tempfile.mkstemp(prefix=".install-status-", dir=str(root))
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(value, out)
        os.replace(name, status_path)
    finally:
        if os.path.exists(name): os.unlink(name)

def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""): h.update(chunk)
    return h.hexdigest()

def staged(item):
    path = root / "pkgs" / item["name"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"] or sha(path) != item["sha256"]:
        raise RuntimeError("Staged file changed or is missing: " + item["name"])
    return path

started = time.time()
retained_prefix = None
save({"status": "installing", "started_at": started})
try:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix_lock = (prefix.parent / ("." + prefix.name + ".dft-agent-lock")).open("a")
    try:
        fcntl.flock(prefix_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError("Another installation is using this folder. Wait for it to finish before retrying.")
    if prefix.is_symlink():
        raise RuntimeError("The installation folder is a symbolic link; choose a new prefix.")
    # Package archives are needed only to create the environment and are removed after a passed test.
    files = {item["name"]: staged(item) for item in plan["downloads"] if item["role"] != "package"}
    receipt = prefix / ".dft-agent-install.json"
    marker = prefix / ".dft-agent-environment.json"
    packages = hashlib.sha256("".join(item["sha256"] for item in plan["downloads"] if item["role"] == "package").encode()).hexdigest()
    owner = prefix.parent / ("." + prefix.name + ".dft-agent-owner.json")
    if receipt.is_file() and json.loads(receipt.read_text()).get("plan_id") == plan["id"]:
        created = False
    elif marker.is_file() and json.loads(marker.read_text()).get("packages") == packages:
        # An earlier attempt created this environment from the same packages; test it again.
        created = False
    else:
        if prefix.exists() and any(prefix.iterdir()):
            owned = json.loads(owner.read_text()) if owner.is_file() and not owner.is_symlink() else {}
            if owned.get("prefix") != str(prefix) or owned.get("packages") != packages:
                raise RuntimeError("The installation folder is not empty and was not created by this installation; choose a new prefix.")
            # Keep a failed partial environment recoverable; never rename a completed environment.
            backup = prefix.with_name(prefix.name + ".incomplete-" + uuid.uuid4().hex[:12])
            prefix.rename(backup)
            retained_prefix = str(backup)
        if owner.is_symlink():
            raise RuntimeError("The installation ownership record is a symbolic link; choose a new prefix.")
        owner.write_text(json.dumps({"prefix": str(prefix), "packages": packages, "plan_id": plan["id"], "created_at": time.time()}))
        files.update({item["name"]: staged(item) for item in plan["downloads"] if item["role"] == "package"})
        archive = next(files[item["name"]] for item in plan["downloads"] if item["role"] == "micromamba")
        binary = root / "bin" / "micromamba"
        binary.parent.mkdir(exist_ok=True)
        with tarfile.open(archive, "r:bz2") as tar:
            member = tar.getmember("bin/micromamba")
            with tar.extractfile(member) as source, binary.open("wb") as target:
                shutil.copyfileobj(source, target)
        binary.chmod(0o755)
        explicit = root / "explicit.txt"
        explicit.write_text("@EXPLICIT\n" + "".join(f"file://{files[item['name']]}#sha256:{item['sha256']}\n"
                                                    for item in plan["downloads"] if item["role"] == "package"))
        environment = {**os.environ, "MAMBA_ROOT_PREFIX": str(root / "mamba"), "CONDA_PKGS_DIRS": str(root / "cache")}
        environment.pop("LD_LIBRARY_PATH", None)
        try:
            created = subprocess.run([str(binary), "create", "-y", "--offline", "-p", str(prefix), "-f", str(explicit)],
                                     env=environment, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, timeout=3600)
        finally:
            # The extracted package cache (gigabytes for CP2K) is not used by the environment.
            shutil.rmtree(str(root / "cache"), ignore_errors=True)
        (root / "micromamba.log").write_text(created.stdout)
        if created.returncode:
            raise RuntimeError("micromamba could not create the environment; read micromamba.log.")
        marker.write_text(json.dumps({"packages": packages, "plan_id": plan["id"], "created_at": time.time()}))
        created = True
    program = prefix / "bin" / plan["program"]
    if not program.is_file() or not os.access(program, os.X_OK):
        raise RuntimeError("The environment has no " + plan["program"] + " program.")
    library = pathlib.Path(plan["library_root"])
    if plan["engine"] == "elk":
        library.mkdir(parents=True, exist_ok=True)
        archive = next(files[item["name"]] for item in plan["downloads"] if item["role"] == "species_archive")
        expected = plan["extras"]["species"]
        with tarfile.open(archive, "r:gz") as tar:
            for name, digest in expected.items():
                member = tar.getmember(plan["extras"]["species_dir"] + "/" + name + ".in")
                if not member.isfile(): raise RuntimeError("Species entry is not a file: " + name)
                data = tar.extractfile(member).read()
                if hashlib.sha256(data).hexdigest() != digest: raise RuntimeError("Species file differs from the pinned release: " + name)
                (library / (name + ".in")).write_bytes(data)
    else:
        library.mkdir(parents=True, exist_ok=True)
        for item in plan["downloads"]:
            if item["role"] == "library":
                target = library / item["name"]
                if not target.exists():
                    shutil.copyfile(files[item["name"]], target)
                if sha(target) != item["sha256"]: raise RuntimeError("Library file differs from the pinned file: " + item["name"])
    smoke = root / "smoke"
    metadata = json.loads((smoke / "metadata.json").read_text())
    keep = {*metadata["input_files"], "metadata.json"}
    for path in smoke.iterdir():
        # A repeated job must not pass on an earlier test's outputs.
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        elif path.name not in keep:
            path.unlink()
    (smoke / metadata["library_dir"]).mkdir(exist_ok=True)
    for name in metadata["library_files"]:
        shutil.copyfile(library / name, smoke / metadata["library_dir"] / name)
    environment = {**os.environ, "OMP_NUM_THREADS": str(plan["build_cpus"] if plan["engine"] == "elk" else 1)}
    environment.pop("LD_LIBRARY_PATH", None)
    if plan.get("mpi") == "mpich":
        environment["HYDRA_BOOTSTRAP"] = "fork"
    limits = None
    if plan["engine"] == "elk":
        import resource
        environment.update(OMP_PROC_BIND="false", OMP_STACKSIZE="512M", OPENBLAS_NUM_THREADS="1")
        hard = resource.getrlimit(resource.RLIMIT_STACK)[1]
        wanted = 512 * 1024 * 1024 if hard == resource.RLIM_INFINITY else min(512 * 1024 * 1024, hard)
        limits = lambda: resource.setrlimit(resource.RLIMIT_STACK, (wanted, hard))
    command = [str(program)] if plan["engine"] == "elk" else [str(prefix / "bin" / "mpiexec"), "-n", str(min(2, plan["build_cpus"])), str(program)]
    begun = time.time()
    with (smoke / metadata["stdout_file"]).open("w") as out:
        result = subprocess.run(command + metadata["arguments"], cwd=str(smoke), env=environment, stdout=out,
                                stderr=subprocess.STDOUT, timeout=1200, preexec_fn=limits)
    elapsed = time.time() - begun
    output = (smoke / metadata["output_file"]).read_text(errors="replace") if (smoke / metadata["output_file"]).is_file() else ""
    marker = {"abinit": "Calculation completed.", "cp2k": "PROGRAM ENDED AT", "elk": "Elk code stopped"}[plan["engine"]]
    if result.returncode or marker not in output:
        log = (smoke / metadata["stdout_file"]).read_text(errors="replace") + output
        needed = sorted(set(re.findall(r"GLIBC_([0-9]+[.][0-9]+)' not found", log)), key=lambda v: tuple(map(int, v.split("."))))
        if needed:
            try:
                node = os.confstr("CS_GNU_LIBC_VERSION").split()[-1]
            except (ValueError, OSError, IndexError):
                node = "an older one"
            raise RuntimeError(plan["program"] + " needs glibc " + needed[-1] + " but the compute node has " + node
                               + "; this package build cannot run here. Use a module the cluster provides.")
        raise RuntimeError(plan["program"] + " did not finish the silicon test (exit " + str(result.returncode) + "); read smoke/" + metadata["output_file"] + ".")
    receipt.write_text(json.dumps({"plan_id": plan["id"], "engine": plan["engine"], "version": plan["version"],
                                   "installed_at": time.time()}))
    for item in plan["downloads"]:
        if item["role"] == "package":
            try:
                (root / "pkgs" / item["name"]).unlink()
            except FileNotFoundError:
                pass
    save({"status": "installed", "prefix": str(prefix), "bin_dir": str(prefix / "bin"), "program": str(program),
          "library_root": str(library), "created": created, "smoke_seconds": round(elapsed, 1),
          "smoke_outputs": sorted(p.name for p in smoke.iterdir() if p.is_file()), "finished_at": time.time()})
except Exception as exc:
    save({"status": "failed", "error": str(exc)[-2000:], "finished_at": time.time()})
    sys.exit(1)
'''


def _job_script(plan):
    from types import SimpleNamespace
    from .config import ClusterConfig
    cpus = plan["build_cpus"]
    namespace = SimpleNamespace(partition=plan["partition"], account=plan["account"], nodes=1,
                                tasks=1 if plan["engine"] == "elk" else min(2, cpus),
                                cpus_per_task=cpus if plan["engine"] == "elk" else 1, walltime=plan["walltime"],
                                sbatch_resource_lines=lambda: [f"#SBATCH --cpus-per-task={cpus}"] if plan["engine"] == "elk" and cpus > 1 else [],
                                pbs_resource_lines=lambda: ClusterConfig.pbs_resource_lines(SimpleNamespace(extra_pbs=plan.get("extra_pbs", []))))
    adapter = get_scheduler(plan["scheduler"])
    lines = adapter.header(namespace, stdout="install.out", stderr="install.err")
    if plan["scheduler"] == "pbs":
        lines.insert(1, "#PBS -N " + adapter.job_name(f"{plan['engine']}-install-{plan['id']}"))
    else:
        lines.insert(1, f"#SBATCH --job-name={plan['engine']}-install-{plan['id']}")
    lines += ["set -euo pipefail", *adapter.working_directory(), "cd " + shlex.quote(plan["work_dir"]), "python3 install.py"]
    return "\n".join(lines) + "\n"


def _download(item, cache, timeout=120):
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / item["name"]
    if target.is_file() and target.stat().st_size == item["size"] and _digest(target) == item["sha256"]:
        return target
    if not item["url"].startswith("https://"):
        raise ValueError("Program downloads require HTTPS.")
    context = ssl.create_default_context()
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    descriptor, temporary = tempfile.mkstemp(prefix=item["name"] + ".", dir=cache)
    try:
        request = urllib.request.Request(item["url"], headers={"User-Agent": "DFT-Agent"})
        with os.fdopen(descriptor, "wb") as out, urllib.request.urlopen(request, timeout=timeout, context=context) as source:
            size = 0
            while block := source.read(1024 * 1024):
                size += len(block)
                if size > item["size"]:
                    raise ValueError("Download exceeded its pinned size.")
                out.write(block)
        if _digest(Path(temporary)) != item["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {item['name']}; nothing was installed.")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


def _stage_status(transport, work):
    code = f"import json,pathlib\np=pathlib.Path({work!r})/'stage-status.json'\nprint({_MARKER!r}+json.dumps(json.loads(p.read_text()) if p.is_file() else {{}}))"
    return _json_run(transport, code)


def start_install(config, plan, transport=None, *, cache_dir=None, wait=900, interval=5.0):
    """Stage the pinned packages and submit the installation job. Repeated calls reuse its receipt."""
    if plan.get("schema_version") != 1 or plan.get("engine") not in ENGINES:
        raise ValueError("Create and review an installation plan first.")
    cache = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".dft-agent" / "downloads" / "conda"
    work = _absolute(plan["work_dir"])
    with _connection(config, transport) as active:
        existing = _json_run(active, f"import json,pathlib\np=pathlib.Path({work!r})/'submission.json'\nprint({_MARKER!r}+json.dumps(json.loads(p.read_text()) if p.is_file() else {{}}))")
        if existing:
            if existing.get("id") == plan["id"] and not existing.get("job_id"):
                return _reconcile(active, plan, existing)
            return existing
        required = get_scheduler(plan["scheduler"]).required_commands
        _json_run(active, f"import json,shutil\nmissing=[n for n in {list(required) + ['python3']!r} if not shutil.which(n)]\n"
                          f"assert not missing, 'Missing commands: '+', '.join(missing)\nprint({_MARKER!r}+json.dumps({{}}))")
        with tempfile.TemporaryDirectory(prefix="dft-engine-install-") as folder:
            folder = Path(folder)
            (folder / "plan.json").write_text(json.dumps(plan, indent=1) + "\n")
            (folder / "stage.py").write_text(_STAGE_PY)
            active.upload_many([folder / "plan.json", folder / "stage.py"], work, permitted_names=frozenset({"plan.json", "stage.py"}))
        launched = active.run(f"cd {shlex.quote(work)} && (nohup python3 stage.py > stage.log 2>&1 < /dev/null &)", timeout=60)
        if launched.returncode:
            raise TransportError((launched.stderr or launched.stdout)[-2000:] or "The package download could not start.")
        deadline = time.monotonic() + wait
        status = {}
        while time.monotonic() < deadline:
            status = _stage_status(active, work)
            if status.get("status") in {"staged", "transfer_required"}:
                break
            time.sleep(interval)
        if status.get("status") not in {"staged", "transfer_required"}:
            return {"status": "staging", "id": plan["id"], "work_dir": work, "progress": status,
                    "next": "The cluster is still downloading packages. Start the installation again to continue."}
        missing = [item for item in plan["downloads"] if item["name"] in set(status.get("failed", []))]
        if missing:
            # The login node could not reach the package server: upload verified local copies.
            paths = [_download(item, cache) for item in missing]
            receipt = active.upload_many(paths, work + "/pkgs", permitted_names=frozenset(item["name"] for item in missing))
            if any(receipt.get(item["name"], {}).get("sha256") != item["sha256"] for item in missing):
                raise TransportError("Uploaded packages returned unexpected checksums. Nothing was submitted.")
        with tempfile.TemporaryDirectory(prefix="dft-engine-install-") as folder:
            folder = Path(folder)
            smoke = folder / "smoke-inputs"
            metadata = _smoke_inputs(plan["engine"], smoke)
            (smoke / "metadata.json").write_text(json.dumps(metadata, indent=1) + "\n")
            names = {"install.py": _INSTALL_PY, "build.sh": _job_script(plan)}
            for name, content in names.items():
                (folder / name).write_text(content, encoding="utf-8", newline="\n")
            uploads = [folder / name for name in names] + [Path(__file__).with_name("dispatch.py"), Path(__file__).with_name("scheduler.py")]
            active.upload_many(uploads, work, permitted_names=frozenset({*names, "dispatch.py", "scheduler.py"}))
            smoke_names = [*metadata["input_files"], "metadata.json"]
            active.upload_many([smoke / name for name in smoke_names], work + "/smoke", permitted_names=frozenset(smoke_names))
        code = f'''import fcntl,json,pathlib,sys,time
root=pathlib.Path({work!r})
with (root/'submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    path=root/'submission.json'
    if path.exists():
        value=json.loads(path.read_text())
    else:
        value={{'status':'submitting','work_dir':str(root),'prefix':{plan['prefix']!r},'id':{plan['id']!r},'engine':{plan['engine']!r},'submitted_at':time.time(),'scheduler':{plan['scheduler']!r}}}
        path.write_text(json.dumps(value)+'\\n')
        sys.path.insert(0,str(root))
        from dispatch import dispatch
        try:
            acknowledged=dispatch(root,{f"{plan['engine']}-install-{plan['id']}"!r},scheduler={plan['scheduler']!r},script='build.sh')
            value.update(status='submitted',job_id=acknowledged['job_id'])
        except Exception as exc:
            value.update(status='needs_attention',error=str(exc)[-2000:])
        temporary=root/'submission.tmp'
        temporary.write_text(json.dumps(value)+'\\n')
        temporary.replace(path)
print({_MARKER!r}+json.dumps(value))
'''
        return _json_run(active, code, setup=config.setup_commands, timeout=120)


def _reconcile(transport, plan, receipt):
    code = f'''import json,pathlib,sys
root=pathlib.Path({plan['work_dir']!r})
sys.path.insert(0,str(root))
value={receipt!r}
try:
    from dispatch import dispatch
    job=dispatch(root,{f"{plan['engine']}-install-{plan['id']}"!r},reconcile_only=True,scheduler={plan['scheduler']!r},script='build.sh')
    value.update(status='submitted',job_id=job['job_id'])
    value.pop('error',None)
except Exception as exc:
    value.update(status='needs_attention',error=str(exc)[-2000:])
temporary=root/'submission-reconciled.tmp'
temporary.write_text(json.dumps(value)+'\\n')
temporary.replace(root/'submission.json')
print({_MARKER!r}+json.dumps(value))
'''
    return _json_run(transport, code, timeout=90)


def poll_install(config, plan, receipt, transport=None):
    """Read the installation job; when it finished, check its silicon test locally."""
    work = _absolute(plan["work_dir"])
    job = str(receipt.get("job_id", "") or "")
    scheduler = get_scheduler(plan["scheduler"])
    if job and not scheduler.valid_job_id(job):
        raise ValueError("Invalid scheduler job ID.")
    code = f'''import json,pathlib,subprocess,sys
root=pathlib.Path({work!r})
status=root/'install-status.json'
stage=root/'stage-status.json'
value=json.loads(status.read_text()) if status.is_file() else {{'status':{receipt.get('status', 'unknown')!r}}}
if stage.is_file(): value['staging']=json.loads(stage.read_text())
job={job!r}
if job and value.get('status') not in ('installed','failed'):
    sys.path.insert(0,str(root))
    from scheduler import get_scheduler, TERMINAL_STATES
    adapter=get_scheduler({plan['scheduler']!r})
    if adapter.name=='pbs':
        query=subprocess.run(adapter.status_command(job),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        phase=adapter.parse_status(query.stdout,job) if query.returncode==0 else 'UNKNOWN'
    else:
        query=subprocess.run(['squeue','-h','-j',job,'-o','%T'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        phase=query.stdout.strip()
        if not phase:
            accounting=subprocess.run(['sacct','-n','-X','-j',job,'--format=State','--parsable2'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
            phase=(accounting.stdout.strip().splitlines() or ['UNKNOWN'])[0]
    value['scheduler_state']=phase
    word=phase.split()[0].rstrip('+') if phase else ''
    if word in TERMINAL_STATES:
        if not status.is_file() or json.loads(status.read_text()).get('status')!='installed':
            value.update(status='failed',error=value.get('error') or 'The installation job ended without a finished receipt. Read install.out and install.err.')
value['work_dir']=str(root)
value['logs']={{name:str(root/name) for name in ('install.out','install.err','micromamba.log','stage.log')}}
print({_MARKER!r}+json.dumps(value))
'''
    with _connection(config, transport) as active:
        value = _json_run(active, code)
        if value.get("status") == "installed" and not value.get("smoke"):
            value["smoke"] = _check_smoke(active, plan, value)
            if not value["smoke"].get("success"):
                value.update(status="failed", error="The silicon test did not pass: " + str(value["smoke"].get("reason")))
    return value


def _check_smoke(transport, plan, status):
    """Download the small silicon test and parse it with the same checks as a project run."""
    from .engine_workflow import adapter
    with tempfile.TemporaryDirectory(prefix="dft-engine-smoke-") as folder:
        folder = Path(folder)
        metadata = _smoke_inputs(plan["engine"], folder / "expected")
        wanted = {*metadata["input_files"], *metadata["output_files"], metadata["stdout_file"]}
        names = [name for name in status.get("smoke_outputs", []) if PurePosixPath(name).name == name and name in wanted]
        manifest = transport.download_many(plan["work_dir"] + "/smoke", folder, names, permitted_names=frozenset(names))
        if set(manifest) != set(names):
            return {"success": False, "reason": "The test outputs could not be downloaded."}
        result = adapter(plan["engine"]).analyze_outputs(folder, "scf")
        return {key: result.get(key) for key in ("success", "reason", "final_energy_ev", "fermi_energy_ev", "converged_electronic")}


def plan_library(engine, elements, *, soc=False, functional="PBE"):
    """Pinned library files an environment needs for these elements."""
    if engine == "abinit":
        from .abinit import catalog, library_key
        key = library_key({"functional": functional, "soc": soc})
        table = catalog()["libraries"][key]
        missing = [element for element in elements if element not in table["elements"]]
        if missing:
            raise ValueError(f"{table['label']} has no pseudopotential for {', '.join(missing)}.")
        return [{"name": table["elements"][e]["filename"], "url": table["elements"][e]["url"], "sha256": table["elements"][e]["sha256"],
                 "size": table["elements"][e]["size"], "element": e, "library": key, "valence": table["elements"][e]["valence"]}
                for e in elements]
    if engine == "cp2k":
        from .cp2k import catalog
        return [{"name": name, "url": value["url"], "sha256": value["sha256"], "size": value["size"], "library": "cp2k-" + catalog()["version"]}
                for name, value in sorted(catalog()["files"].items())]
    if engine == "elk":
        from .elk import catalog
        table = catalog()["species"]
        missing = [element for element in elements if element not in table]
        if missing:
            raise ValueError(f"Elk {catalog()['version']} has no species file for {', '.join(missing)}.")
        return [{"name": table[e]["filename"], "sha256": table[e]["sha256"], "size": table[e]["size"], "element": e,
                 "library": "elk-" + catalog()["version"]} for e in (elements or sorted(table))]
    raise ValueError("Choose ABINIT, CP2K or Elk.")


def ensure_library(config, elements, transport=None, *, cache_dir=None, soc=False, functional="PBE"):
    """Install pinned library files into the environment's library folder, checking every hash."""
    engine = config.engine
    items = plan_library(engine, list(elements), soc=soc, functional=functional)
    root = _absolute(config.engine_library_root)
    cache = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".dft-agent" / "downloads" / engine
    archive = None
    if engine == "elk":
        from .elk import catalog
        archive = {key: catalog()["archive"][key] for key in ("url", "sha256", "size", "filename", "species_dir")}
    code = f'''import hashlib,io,json,os,pathlib,tarfile,tempfile,urllib.request
root=pathlib.Path({root!r})
root.mkdir(parents=True,exist_ok=True)
items={[{k: item.get(k) for k in ("name", "url", "sha256", "size")} for item in items]!r}
archive={archive!r}
result={{}}
def fetch(url,size):
    request=urllib.request.Request(url,headers={{'User-Agent':'DFT-Agent'}})
    return urllib.request.urlopen(request,timeout=30).read(size+1)
def save(name,data):
    fd,temporary=tempfile.mkstemp(prefix='.library-',dir=str(root))
    with os.fdopen(fd,'wb') as out: out.write(data)
    os.replace(temporary,root/name)
pending=[]
for item in items:
    target=root/item['name']
    if target.is_file() and not target.is_symlink() and hashlib.sha256(target.read_bytes()).hexdigest()==item['sha256']:
        result[item['name']]='ready'
    else:
        pending.append(item)
if pending and archive:
    try:
        data=fetch(archive['url'],archive['size'])
        if len(data)!=archive['size'] or hashlib.sha256(data).hexdigest()!=archive['sha256']: raise ValueError('checksum')
        with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
            for item in pending:
                member=tar.extractfile(archive['species_dir']+'/'+item['name'])
                content=member.read() if member else b''
                if hashlib.sha256(content).hexdigest()!=item['sha256']: raise ValueError('species checksum')
                save(item['name'],content)
                result[item['name']]='ready'
    except Exception:
        pass
for item in pending:
    if item['name'] in result: continue
    try:
        if not item.get('url'): raise ValueError('no direct URL')
        data=fetch(item['url'],item['size'])
        if len(data)!=item['size'] or hashlib.sha256(data).hexdigest()!=item['sha256']: raise ValueError('checksum')
        save(item['name'],data)
        result[item['name']]='ready'
    except Exception:
        result[item['name']]='transfer_required'
print({_MARKER!r}+json.dumps(result))
'''
    with _connection(config, transport) as active:
        statuses = _json_run(active, code, timeout=max(120, 35 * len(items)))
        missing = [item for item in items if statuses.get(item["name"]) != "ready"]
        if missing:
            paths = _local_library(engine, missing, cache, archive)
            receipt = active.upload_many(paths, root, permitted_names=frozenset(item["name"] for item in missing))
            if any(receipt.get(item["name"], {}).get("sha256") != item["sha256"] for item in missing):
                raise TransportError("Uploaded library files returned unexpected checksums.")
    return {"library_root": root, "files": {item["name"]: {key: item[key] for key in ("sha256", "element", "library", "valence") if key in item}
                                            for item in items}}


def _local_library(engine, items, cache, archive):
    """Verified local copies for files the cluster could not download itself."""
    if engine != "elk":
        return [_download(item, cache) for item in items]
    import tarfile
    source = _download({"name": archive["filename"], "url": archive["url"], "sha256": archive["sha256"], "size": archive["size"]}, cache)
    folder = cache / "species"
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    with tarfile.open(source, "r:gz") as tar:
        for item in items:
            member = tar.extractfile(archive["species_dir"] + "/" + item["name"])
            data = member.read() if member else b""
            if hashlib.sha256(data).hexdigest() != item["sha256"]:
                raise ValueError(f"The Elk species file {item['name']} differs from the pinned release.")
            (folder / item["name"]).write_bytes(data)
            paths.append(folder / item["name"])
    return paths


def doctor(config, transport=None):
    """Check the scheduler, the launch command's program and the library folder without running a job."""
    import shlex as _shlex
    from .engine_workflow import PROGRAMS
    tokens = _shlex.split(config.engine_command.replace("{tasks}", "1"))
    program = next((token for token in tokens if PurePosixPath(token).name in PROGRAMS[config.engine]), "")
    code = f"""import json,os,shutil
program={program!r}
found=shutil.which(program) or ''
root={config.engine_library_root!r}
print({_MARKER!r}+json.dumps({{'program':found,
    'library':os.path.isdir(root),'tools':{{n:bool(shutil.which(n)) for n in {list(get_scheduler(config).required_commands)!r}}}}}))
"""
    try:
        with _connection(config, transport) as active:
            state = _json_run(active, code, setup=config.setup_commands, timeout=90)
    except (TransportError, OSError, ValueError) as exc:
        return {"ok": False, "checks": {"connection": {"ok": False, "detail": str(exc)}}}
    checks = {LABELS[config.engine]: {"ok": bool(state["program"]), "detail": state["program"] or f"{program or 'program'} not found"}}
    for name, present in state["tools"].items():
        checks[name] = {"ok": present, "detail": "found" if present else "Not found"}
    checks["library"] = {"ok": state["library"], "detail": config.engine_library_root if state["library"]
                         else "Install the library files from the app or with 'dft-agent engine library'."}
    return {"ok": all(check["ok"] for check in checks.values()), "checks": checks}
