"""Versioned QE builds and public pseudopotentials for Slurm clusters."""

from __future__ import annotations

import contextlib
from copy import deepcopy
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import ssl
import tarfile
import tempfile
import urllib.request
import uuid

from .transport import SSHTransport, TransportError

QE_VERSION = "7.4.1"
REQUIRED_PROGRAMS = ("pw.x", "bands.x", "dos.x", "projwfc.x", "pp.x", "ph.x", "q2r.x", "matdyn.x", "dynmat.x", "neb.x", "lambda.x")
BUILD_TARGETS = ("pw", "ph", "pp", "neb")
EPW_PROGRAMS = ("epw.x", "wannier90.x")
EPW_SOURCES = {
    "wannier90.tar.gz": {
        "url": "https://codeload.github.com/wannier-developers/wannier90/tar.gz/1d6b187374a2d50b509e5e79e2cab01a79ff7ce1",
        "sha256": "bc93ba6774ecc41aa766303747ef355849d63f469c2a437d03acda4e98515725",
        "source_archive_sha256": "e5936d20c4e19f458335a53a1ec94ac993e54b09ee0e44b3184ac13c0f2758ad",
        "include_paths": ["src", "config", "Makefile", "LICENSE", "README.rst", "README.install", "CHANGELOG.md"],
    },
}
SOURCES = {
    "qe-7.4.1.tar.gz": {
        "url": "https://gitlab.com/QEF/q-e/-/archive/qe-7.4.1/q-e-qe-7.4.1.tar.gz",
        "sha256": "6ef9c53dbf0add2a5bf5ad2a372c0bff935ad56c4472baa001003e4f932cab97",
    },
    "devxlib.tar.gz": {
        "url": "https://gitlab.com/max-centre/components/devicexlib/-/archive/a6b89ef77b1ceda48e967921f1f5488d2df9226d/devicexlib-a6b89ef77b1ceda48e967921f1f5488d2df9226d.tar.gz",
        "sha256": "39d55ae63168f9692ab649e7cededb3bc181580f450d30d5d4deeb008ea4a29c",
    },
    "fox.tar.gz": {
        "url": "https://codeload.github.com/pietrodelugas/fox/tar.gz/3453648e6837658b747b895bb7bef4b1ed2eac40",
        "sha256": "107b1d58eb6a0a730c98520a8f99a6459b2a0e06b06f4b037abf358b29d50df1",
    },
    "lapack.tar.gz": {
        "url": "https://codeload.github.com/Reference-LAPACK/lapack/tar.gz/12d825396fcef1e0a1b27be9f119f9e554621e55",
        "sha256": "2ae8b6eb09e9c1d3b5d66bb778a5a5119af13575bb7fd8016f5cabb319253257",
    },
    "mbd.tar.gz": {
        "url": "https://codeload.github.com/libmbd/libmbd/tar.gz/89a3cc199c0a200c9f0f688c3229ef6b9a8d63bd",
        "sha256": "956eaa8659a24e59573a64e78372cee9f299b4bf0a89a82f6bf98685034c7736",
    },
}
SSSP_URL = "https://archive.materialscloud.org/records/rcyfm-68h65/files/"
PSEUDO_ARCHIVE = {"url": SSSP_URL + "SSSP_1.3.0_PBE_efficiency.tar.gz", "sha256": "7a85b71fa3d68df1b5ed33c55c7057681fbf10377ae3b3eac9392924d9189f12"}
PSEUDO_LICENSE = {"url": SSSP_URL + "LICENSE.txt", "sha256": "8028d9d7fcad7932bb5551dc0a60cff6c6b7f7aa2bd480f516a2faaa09d943a1"}
_MARKER = "__DFT_QE_SETUP__"


def _absolute(value: str) -> str:
    if not isinstance(value, str) or not value.startswith("/") or any(c in value for c in "\r\n\0") or ".." in PurePosixPath(value).parts:
        raise ValueError("Use an absolute remote path without '..' or line breaks.")
    if value.rstrip("/") in ("", "/usr", "/usr/local", "/opt", "/bin", "/home"):
        raise ValueError("Choose a dedicated folder in your own cluster storage.")
    return value.rstrip("/")


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _build_source_archive(source: Path, target: Path, metadata: dict):
    """Keep pinned build files; large reference datasets stay in the source cache."""
    if _digest(source) != metadata["source_archive_sha256"]:
        raise ValueError("The original build source checksum changed.")
    files = {}
    with tarfile.open(source, "r:gz") as archive:
        for member in archive.getmembers():
            parts = PurePosixPath(member.name).parts
            if len(parts) < 2 or parts[1] not in metadata["include_paths"] or member.isdir():
                continue
            if member.name.startswith("/") or ".." in parts or not member.isfile():
                raise ValueError("Unsafe build source member.")
            name = PurePosixPath(*parts[1:]).as_posix()
            if name in files:
                raise ValueError("Duplicate build source member.")
            files[name] = (archive.extractfile(member).read(), 0o755 if member.mode & 0o111 else 0o644)
    manifest = {"source_url": metadata["url"], "source_archive_sha256": metadata["source_archive_sha256"],
                "include_paths": metadata["include_paths"],
                "excluded": "Reference datasets, pseudopotentials, documentation and unused utilities.",
                "files": {name: hashlib.sha256(data).hexdigest() for name, (data, _) in files.items()}}
    files[".dft-agent-source.json"] = ((json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode(), 0o644)
    with target.open("wb") as stream, gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for name, (data, mode) in sorted(files.items()):
                member = tarfile.TarInfo("source/" + name)
                member.size, member.mode = len(data), mode
                archive.addfile(member, io.BytesIO(data))


def _download(name: str, metadata: dict, cache: Path) -> Path:
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / name
    if target.is_file() and _digest(target) == metadata["sha256"]:
        return target
    if not metadata["url"].startswith("https://"):
        raise ValueError("QE downloads require HTTPS.")
    if "source_archive_sha256" in metadata:
        source = _download("upstream-" + name, {"url": metadata["url"], "sha256": metadata["source_archive_sha256"]}, cache)
        descriptor, temporary = tempfile.mkstemp(prefix=name + ".", dir=cache)
        os.close(descriptor)
        try:
            _build_source_archive(source, Path(temporary), metadata)
            if _digest(Path(temporary)) != metadata["sha256"]:
                raise ValueError(f"Derived build source checksum mismatch for {name}.")
            os.replace(temporary, target)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return target
    context = ssl.create_default_context()
    try:
        import certifi
        context.load_verify_locations(certifi.where())
    except ImportError:
        pass
    descriptor, temporary = tempfile.mkstemp(prefix=name + ".", dir=cache)
    try:
        with os.fdopen(descriptor, "wb") as out, urllib.request.urlopen(metadata["url"], timeout=120, context=context) as source:
            if not source.geturl().startswith("https://"):
                raise ValueError("Download redirected outside HTTPS.")
            size = 0
            while block := source.read(1024 * 1024):
                size += len(block)
                if size > 512 * 1024 * 1024:
                    raise ValueError("Download exceeded the expected size limit.")
                out.write(block)
        if _digest(Path(temporary)) != metadata["sha256"]:
            raise ValueError(f"SHA-256 mismatch for {name}; the file was not installed.")
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return target


@contextlib.contextmanager
def _connection(config, transport):
    active = transport or SSHTransport(config)
    try:
        yield active
    finally:
        if transport is None:
            active.close()


def _json_run(transport, code: str, *, setup=(), timeout=60):
    command = "\n".join(["set -e", *setup, "python3 -c " + shlex.quote(code)])
    result = transport.run(command, timeout=timeout)
    if result.returncode:
        raise TransportError((result.stderr or result.stdout).strip()[-2000:] or "QE setup failed.")
    for line in reversed(result.stdout.splitlines()):
        if line.startswith(_MARKER):
            return json.loads(line[len(_MARKER):])
    raise TransportError("QE setup returned no status. Check the cluster environment.")


def inspect(config, transport=None, *, include_epw=False) -> dict:
    """Check paths and prerequisites without launching a calculation."""
    bindir = getattr(config, "qe_bin_dir", "")
    if bindir:
        _absolute(bindir)
    programs = REQUIRED_PROGRAMS + (EPW_PROGRAMS if include_epw else ())
    code = f'''import json, os, shutil, subprocess
tools = {{name: shutil.which(name) for name in ["python3", "sbatch", "squeue", "sacct", "make", "git", "mpif90", "mpifort", "mpiifort", "cc", "gcc", "icc"]}}
programs = {{}}
for name in {programs!r}:
    path = os.path.join({bindir!r}, name) if {bool(bindir)!r} else shutil.which(name)
    executable = bool(path and os.path.isfile(path) and os.access(path, os.X_OK))
    missing = []
    if executable and shutil.which("ldd"):
        check = subprocess.run(["ldd", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=15)
        missing = [line.strip() for line in check.stdout.splitlines() if "not found" in line]
    programs[name] = {{"path": path, "ok": executable and not missing, "missing_libraries": missing}}
root = {getattr(config, 'qe_pseudo_root', '')!r}
print({_MARKER!r} + json.dumps({{"tools": tools, "programs": programs, "pseudo_root": root, "pseudo_root_exists": bool(root and os.path.isdir(root))}}))
'''
    with _connection(config, transport) as active:
        return _json_run(active, code, setup=config.setup_commands)


def doctor(config, transport=None, *, include_epw=False) -> dict:
    try:
        state = inspect(config, transport, include_epw=include_epw)
    except (TransportError, OSError, ValueError) as exc:
        return {"ok": False, "checks": {"connection": {"ok": False, "detail": str(exc)}}}
    checks = {name: {"ok": item["ok"], "detail": "; ".join(item["missing_libraries"]) or item["path"] or "Not found"} for name, item in state["programs"].items()}
    for name in ("sbatch", "squeue", "sacct"):
        checks[name] = {"ok": bool(state["tools"][name]), "detail": state["tools"][name] or "Not found"}
    checks["pseudopotentials"] = {"ok": state["pseudo_root_exists"], "detail": state["pseudo_root"] or "Choose a UPF folder or install the public library."}
    return {"ok": all(check["ok"] for check in checks.values()), "checks": checks}


def install_plan(config, prefix=None, *, build_jobs=8, build_walltime="02:00:00", include_epw=False) -> dict:
    """Return a reviewable CPU/MPI build plan. Nothing is downloaded or submitted."""
    if type(build_jobs) is not int or not 1 <= build_jobs <= 64:
        raise ValueError("Use 1 to 64 build CPUs.")
    if not re.fullmatch(r"(?:\d+-)?\d{1,3}:[0-5]\d:[0-5]\d", build_walltime):
        raise ValueError("Use HH:MM:SS for the build time limit.")
    if type(include_epw) is not bool:
        raise ValueError("Choose whether to include EPW.")
    base = _absolute(config.remote_root) + "/software"
    plan = {"version": QE_VERSION, "backend": "cpu", "sources": deepcopy(SOURCES), "setup_commands": list(config.setup_commands), "partition": config.partition, "account": config.account, "build_jobs": build_jobs, "walltime": build_walltime, "programs": list(REQUIRED_PROGRAMS), "build_targets": list(BUILD_TARGETS)}
    if include_epw:
        plan.update(include_epw=True, sources=deepcopy(SOURCES | EPW_SOURCES),
                    programs=list(REQUIRED_PROGRAMS + EPW_PROGRAMS),
                    build_targets=[*BUILD_TARGETS, "epw"],
                    build_steps=[list(BUILD_TARGETS), ["epw"]])
    signature = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:12]
    plan["prefix"] = _absolute(prefix) if prefix else base + f"/qe-{QE_VERSION}-{signature}"
    plan["id"] = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16]
    plan["work_dir"] = base + "/builds/" + plan["id"]
    plan["bin_dir"] = plan["prefix"] + "/bin"
    return plan


# This script runs only in a Slurm allocation. Dependencies are already downloaded
# and pinned, so compute nodes do not need Internet access.
_BUILD_PY = r'''
import hashlib, json, os, pathlib, shutil, subprocess, sys, tarfile, time
root = pathlib.Path(__file__).resolve().parent
plan = json.loads((root / 'plan.json').read_text())
def save(value):
    temporary = root / 'build-status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(root / 'build-status.json')
def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): h.update(block)
    return h.hexdigest()
def unpack(path, target):
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, 'r:gz') as archive:
        members = archive.getmembers()
        tops = {pathlib.PurePosixPath(m.name).parts[0] for m in members if m.name}
        if len(tops) != 1: raise RuntimeError('Unexpected archive layout')
        links = []
        for m in members:
            parts = pathlib.PurePosixPath(m.name).parts
            if not parts or m.name.startswith('/') or '..' in parts: raise RuntimeError('Unsafe archive path')
            relative = pathlib.Path(*parts[1:])
            if not relative.parts: continue
            dest = target / relative
            if m.isdir(): dest.mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                dest.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(m) as src, dest.open('xb') as out: shutil.copyfileobj(src, out)
                dest.chmod(m.mode & 0o777)
            elif m.issym(): links.append((dest, m.linkname))
            else: raise RuntimeError('Unsupported archive member')
        for dest, link in links:
            resolved = (dest.parent / link).resolve()
            if not str(resolved).startswith(str(target.resolve()) + os.sep): raise RuntimeError('Unsafe archive link')
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(link)
started = time.time()
prefix = pathlib.Path(plan['prefix'])
stage = root / 'install.partial'
try:
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Builds must run in Slurm')
    targets = plan.get('build_targets')
    include_epw = plan.get('include_epw', False)
    expected_targets = ['pw', 'ph', 'pp', 'neb'] + (['epw'] if include_epw else [])
    if type(include_epw) is not bool or targets != expected_targets: raise RuntimeError('Unsupported QE build targets')
    expected_steps = [['pw', 'ph', 'pp', 'neb'], ['epw']] if include_epw else [targets]
    if plan.get('build_steps', [targets]) != expected_steps: raise RuntimeError('Unsupported QE build sequence')
    if prefix.exists() or prefix.is_symlink(): raise RuntimeError('Installation folder already exists; it was left unchanged')
    save({'status': 'building', 'started_at': started, 'job_id': os.environ['SLURM_JOB_ID']})
    for name, item in plan['sources'].items():
        if digest(root / name) != item['sha256']: raise RuntimeError('Source checksum mismatch: ' + name)
    source = root / 'source'
    if source.exists() or stage.exists(): raise RuntimeError('Build folder is not empty; use a new installation prefix')
    unpack(root / 'qe-7.4.1.tar.gz', source)
    dependencies = ('devxlib', 'fox', 'lapack', 'mbd') + (('wannier90',) if include_epw else ())
    for name in dependencies:
        folder = source / 'external' / name
        unpack(root / (name + '.tar.gz'), folder)
        source_metadata = plan['sources'][name + '.tar.gz']
        if 'source_archive_sha256' in source_metadata:
            record = json.loads((folder / '.dft-agent-source.json').read_text())
            if record['source_archive_sha256'] != source_metadata['source_archive_sha256']: raise RuntimeError('Original source identity changed: ' + name)
            actual = {p.relative_to(folder).as_posix(): digest(p) for p in folder.rglob('*') if p.is_file() and p.name != '.dft-agent-source.json'}
            if actual != record['files']: raise RuntimeError('Extracted source manifest mismatch: ' + name)
        (folder / '.git').touch()  # QE accepts an already staged, pinned dependency.
    mpi = os.environ.get('MPIF90') or next((x for x in ('mpiifort', 'mpifort', 'mpif90') if shutil.which(x)), None)
    if not mpi: raise RuntimeError('Load an MPI Fortran compiler before building QE')
    environment = os.environ.copy()
    environment['OMP_NUM_THREADS'] = '1'
    command = ['./configure', '--enable-parallel', '--prefix=' + str(stage), 'MPIF90=' + mpi]
    subprocess.run(command, cwd=source, env=environment, check=True)
    make_inc = (source / 'make.inc').read_text()
    if '__MPI' not in make_inc: raise RuntimeError('Configure did not enable MPI')
    # EPW recursively builds PH; finish the core first to avoid concurrent writes.
    for step in expected_steps:
        subprocess.run(['make', '-j', str(plan['build_jobs']), *step], cwd=source, env=environment, check=True)
    subprocess.run(['make', 'install'], cwd=source, env=environment, check=True)
    programs = {}
    for name in plan['programs']:
        path = stage / 'bin' / name
        if not path.is_file() or not os.access(path, os.X_OK): raise RuntimeError('Missing executable: ' + name)
        if shutil.which('ldd'):
            check = subprocess.run(['ldd', str(path)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            if 'not found' in check.stdout: raise RuntimeError('Missing runtime library for ' + name)
        programs[name] = digest(path)
    for name in ('License', 'README.md'):
        if (source / name).is_file(): shutil.copy2(source / name, stage / name)
    manifest = dict(plan, installed_at=time.time(), job_id=os.environ['SLURM_JOB_ID'], executable_sha256=programs, compiler=mpi, make_inc=make_inc)
    (stage / 'dft-agent-install.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if prefix.exists() or prefix.is_symlink(): raise RuntimeError('Installation folder appeared during build; refusing replacement')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    stage.rename(prefix)
    save({'status': 'installed', 'job_id': os.environ['SLURM_JOB_ID'], 'prefix': str(prefix), 'bin_dir': str(prefix / 'bin'), 'programs': programs, 'elapsed_seconds': time.time()-started})
except Exception as exc:
    save({'status': 'failed', 'job_id': os.environ.get('SLURM_JOB_ID'), 'error': str(exc), 'elapsed_seconds': time.time()-started})
    raise
'''


def _build_script(plan: dict) -> str:
    lines = ["#!/bin/bash", f"#SBATCH --job-name=qe-build-{plan['id']}", f"#SBATCH --partition={plan['partition']}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1", f"#SBATCH --cpus-per-task={plan['build_jobs']}", f"#SBATCH --time={plan['walltime']}", "#SBATCH --output=build.out", "#SBATCH --error=build.err"]
    if plan["account"]:
        lines.append(f"#SBATCH --account={plan['account']}")
    lines.extend(["set -euo pipefail", *plan["setup_commands"], "export OMP_NUM_THREADS=1", "cd " + shlex.quote(plan["work_dir"]), "python3 build.py"])
    return "\n".join(lines) + "\n"


def _source_transfers(transport, sources, work, cache):
    """Use verified cluster downloads first; return only files needing transfer."""
    code = f'''import hashlib,json,multiprocessing,os,pathlib,signal,tempfile,urllib.request
root=pathlib.Path({work!r})
root.mkdir(parents=True,exist_ok=True)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()
def download(url,temporary,pipe):
    try:
        request=urllib.request.Request(url,headers={{'User-Agent':'DFT-Agent'}})
        with open(temporary,'wb') as out,urllib.request.urlopen(request,timeout=10) as response:
            if not response.geturl().startswith('https://'): raise RuntimeError('HTTPS redirect required')
            size=0
            while True:
                data=response.read(1048576)
                if not data: break
                size+=len(data)
                if size>536870912: raise RuntimeError('Source archive exceeds size limit')
                out.write(data)
        pipe.send(('ready',''))
    except RuntimeError as exc:
        pipe.send(('invalid',str(exc)))
    except Exception:
        pipe.send(('transfer_required',''))
    finally:
        pipe.close()
result={{}}
for name,metadata in {sources!r}.items():
    p=root/name
    if p.exists() or p.is_symlink():
        if p.is_symlink() or not p.is_file() or sha(p)!=metadata['sha256']:
            raise RuntimeError('Existing source has a different checksum; choose a new installation prefix')
        result[name]='ready'
        continue
    if 'source_archive_sha256' in metadata:
        result[name]='transfer_required'
        continue
    if not metadata['url'].startswith('https://'): raise RuntimeError('QE downloads require HTTPS')
    fd,temporary=tempfile.mkstemp(prefix='.qe-download-',dir=str(root))
    temp=pathlib.Path(temporary)
    try:
        os.close(fd)
        receive,send=multiprocessing.get_context('fork').Pipe(duplex=False)
        process=multiprocessing.get_context('fork').Process(target=download,args=(metadata['url'],str(temp),send))
        process.start()
        send.close()
        process.join(20)
        if process.is_alive():
            process.terminate()
            process.join(2)
            if process.is_alive():
                os.kill(process.pid,signal.SIGKILL)
                process.join(2)
            status,detail='transfer_required',''
        else:
            status,detail=receive.recv() if receive.poll() else ('transfer_required','')
        receive.close()
        if status=='invalid': raise RuntimeError(detail)
        if status!='ready':
            result[name]='transfer_required'
            continue
        if sha(temp)!=metadata['sha256']: raise RuntimeError('Downloaded source checksum mismatch')
        try: os.link(str(temp),str(p))
        except FileExistsError:
            if p.is_symlink() or not p.is_file() or sha(p)!=metadata['sha256']: raise RuntimeError('Source changed during download')
        result[name]='ready'
    except RuntimeError:
        raise
    except Exception:
        result[name]='transfer_required'
    finally:
        if temp.exists(): temp.unlink()
print({_MARKER!r}+json.dumps(result))
'''
    statuses = _json_run(transport, code, timeout=max(180, 35 * len(sources)))
    if set(statuses) != set(sources) or any(status not in {"ready", "transfer_required"} for status in statuses.values()):
        raise TransportError("QE source staging returned an invalid receipt.")
    return [_download(name, sources[name], cache) for name, status in statuses.items() if status == "transfer_required"]


def start_install(config, plan: dict, transport=None, *, cache_dir=None) -> dict:
    """Stage verified source and submit one build. Repeated calls reuse its receipt."""
    expected = install_plan(config, prefix=plan["prefix"], build_jobs=plan["build_jobs"], build_walltime=plan["walltime"], include_epw=plan.get("include_epw", False))
    if plan != expected:
        raise ValueError("The installation plan changed. Create and review a new plan.")
    cache = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".dft-agent" / "downloads" / "qe"
    work = plan["work_dir"]
    with _connection(config, transport) as active:
        existing = _json_run(active, f"import json,pathlib\np=pathlib.Path({work!r})/'submission.json'\nprint({_MARKER!r}+json.dumps(json.loads(p.read_text()) if p.is_file() else {{}}))")
        if existing:
            return existing
        state = inspect(config, active)
        if not state["tools"]["sbatch"] or not state["tools"]["make"] or not state["tools"]["git"] or not any(state["tools"][x] for x in ("mpif90", "mpifort", "mpiifort")):
            raise ValueError("Load make, Git and an MPI Fortran compiler; Slurm must be available.")
        transfers = _source_transfers(active, plan["sources"], work, cache)
        staging = work + "/.stage-" + uuid.uuid4().hex
        with tempfile.TemporaryDirectory(prefix="dft-qe-build-") as folder:
            folder = Path(folder)
            for name, content in {"plan.json": json.dumps(plan, indent=2) + "\n", "build.py": _BUILD_PY, "build.sh": _build_script(plan)}.items():
                path = folder / name
                path.write_text(content, encoding="utf-8", newline="\n")
                transfers.append(path)
            expected_files = {path.name: {"sha256": _digest(path), "size": path.stat().st_size} for path in transfers}
            transferred = active.upload_many(transfers, staging, permitted_names=frozenset({*plan["sources"], "plan.json", "build.py", "build.sh"}))
            if transferred != expected_files:
                raise TransportError("QE source upload returned unexpected checksums. No build was submitted.")
        code = f'''import fcntl,hashlib,json,os,pathlib,shutil,subprocess,time
root=pathlib.Path({work!r})
staged=pathlib.Path({staging!r})
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1048576),b''): h.update(chunk)
    return h.hexdigest()
def save(path,value):
    temporary=root/'submission.tmp'
    temporary.write_text(json.dumps(value)+'\\n')
    temporary.replace(path)
with (root/'submit.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX)
    path=root/'submission.json'
    if path.exists():
        value=json.loads(path.read_text())
        shutil.rmtree(staged)
    else:
        try:
            for name,item in {expected_files!r}.items():
                source=staged/name
                if source.is_symlink() or not source.is_file() or source.stat().st_size!=item['size'] or sha(source)!=item['sha256']:
                    raise RuntimeError('Staged installer file changed: '+name)
                target=root/name
                if target.exists() or target.is_symlink():
                    if target.is_symlink() or not target.is_file() or sha(target)!=item['sha256']:
                        raise RuntimeError('Existing installer file differs; choose a new installation prefix: '+name)
            for name,item in {expected_files!r}.items():
                try: os.link(str(staged/name),str(root/name))
                except FileExistsError:
                    if (root/name).is_symlink() or sha(root/name)!=item['sha256']: raise RuntimeError('Installer file changed: '+name)
        finally:
            shutil.rmtree(staged)
        for name,item in {plan['sources']!r}.items():
            source=root/name
            if source.is_symlink() or not source.is_file() or sha(source)!=item['sha256']:
                raise RuntimeError('Source checksum changed before submission: '+name)
        value={{'status':'submitting','work_dir':str(root),'prefix':{plan['prefix']!r},'id':{plan['id']!r},'submitted_at':time.time()}}
        save(path,value)
        result=subprocess.run(['sbatch','--parsable','--chdir',str(root),str(root/'build.sh')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        job_id=result.stdout.strip().split(';')[0]
        if result.returncode == 0 and job_id.isdigit():
            value.update(status='submitted',job_id=job_id)
        else:
            value.update(status='needs_attention',error=(result.stderr or result.stdout).strip()[-2000:])
        save(path,value)
print({_MARKER!r}+json.dumps(value))
'''
        return _json_run(active, code, setup=config.setup_commands, timeout=90)


def poll_install(config, receipt: dict, transport=None) -> dict:
    work = _absolute(receipt["work_dir"])
    job = str(receipt.get("job_id", ""))
    if job and not job.isdigit():
        raise ValueError("Invalid Slurm job ID.")
    code = f'''import json,pathlib,subprocess
root=pathlib.Path({work!r})
status=root/'build-status.json'
value=json.loads(status.read_text()) if status.is_file() else {{'status':{receipt.get('status', 'unknown')!r}}}
if {bool(job)!r} and value.get('status') not in ('installed','failed'):
    job={job!r}
    query=subprocess.run(['squeue','-h','-j',job,'-o','%T'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
    states=query.stdout.strip()
    if states:
        value['scheduler_state']=states
    else:
        accounting=subprocess.run(['sacct','-n','-X','-j',job,'--format=State','--parsable2'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
        states=accounting.stdout.strip()
        value['scheduler_state']=states or 'unknown'
        if any(x in states for x in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','COMPLETED')):
            value.update(status='failed',error='Build ended without an installed manifest. Read build.out and build.err.')
value['work_dir']=str(root)
value['logs']={{name:str(root/name) for name in ('build.out','build.err')}}
print({_MARKER!r}+json.dumps(value))
'''
    with _connection(config, transport) as active:
        return _json_run(active, code, setup=config.setup_commands)


def plan_pseudopotentials(elements, *, soc=False, library="sssp") -> dict:
    """Select an explicit public PBE library and its cutoff hints."""
    if library not in {"sssp", "pseudodojo"}:
        raise ValueError("Choose SSSP or PseudoDojo.")
    if soc:
        raise ValueError("SOC needs fully relativistic UPFs. Select your own verified library; these public libraries are scalar relativistic.")
    if library == "pseudodojo":
        catalogue = json.loads(Path(__file__).with_name("qe_nc_pseudos.json").read_text())
        selected = {}
        for element in dict.fromkeys(elements):
            if element not in catalogue["elements"]:
                raise ValueError(f"No PseudoDojo pseudopotential for {element}. Provide a compatible NC UPF.")
            selected[element] = dict(catalogue["elements"][element], license=catalogue["license"]["name"],
                                     license_url=catalogue["license"]["url"])
        if not selected:
            raise ValueError("Select at least one element.")
        return {"library": catalogue["library"], "library_id": library,
                "source_url": catalogue["archive"]["url"], "archive_sha256": catalogue["archive"]["sha256"],
                "license": catalogue["license"], "cutoff_source": catalogue["cutoff_source"],
                "citations": catalogue["citations"], "functional": "PBE", "relativistic": "scalar",
                "pseudo_type": "NC", "elements": selected,
                "ecutwfc_ry": max(v["cutoff_wfc"] for v in selected.values()),
                "ecutrho_ry": max(v["cutoff_rho"] for v in selected.values())}
    catalogue = json.loads(Path(__file__).with_name("qe_pseudos.json").read_text())
    selected = {}
    for element in dict.fromkeys(elements):
        if element not in catalogue:
            raise ValueError(f"No SSSP pseudopotential for {element}. Provide a compatible UPF.")
        item = dict(catalogue[element])
        family = item["pseudopotential"]
        item["license"] = "GPL-3.0" if family.startswith(("GBRV", "Wentzcovitch")) else "GPL-2.0-or-later" if family.endswith(("US", "PAW")) else "See SSSP-LICENSE.txt"
        item["license_url"] = PSEUDO_LICENSE["url"]
        selected[element] = item
    if not selected:
        raise ValueError("Select at least one element.")
    return {"library": "SSSP 1.3.0 PBE efficiency", "source_url": PSEUDO_ARCHIVE["url"], "archive_sha256": PSEUDO_ARCHIVE["sha256"], "functional": "PBE", "relativistic": "scalar", "elements": selected, "ecutwfc_ry": max(v["cutoff_wfc"] for v in selected.values()), "ecutrho_ry": max(v["cutoff_rho"] for v in selected.values())}


def validate_upf(data: bytes, element: str) -> dict:
    """Read UPF 1/2 headers, not numeric arrays or external XML entities."""
    if len(data) < 200 or len(data) > 50 * 1024 * 1024:
        raise ValueError("UPF file is empty or unexpectedly large.")
    text = data.decode("utf-8", errors="replace")
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("UPF external entities are not supported.")
    match = re.search(r"<PP_HEADER\b([^>]*)(?:/\s*>|>(.*?)</PP_HEADER>)", text, re.S)
    if not match or "<PP_MESH" not in text or "<PP_LOCAL" not in text:
        raise ValueError("Not a complete UPF file.")
    attributes = dict(re.findall(r'([\w_]+)\s*=\s*[\"\']([^\"\']*)[\"\']', match[1]))
    if attributes:
        actual = attributes.get("element", "").strip()
        functional = attributes.get("functional", "").strip()
        pseudo_type = attributes.get("pseudo_type", "").strip().upper()
        if attributes.get("is_paw", "").lower() in ("t", "true", ".true."):
            pseudo_type = "PAW"
        elif attributes.get("is_ultrasoft", "").lower() in ("t", "true", ".true."):
            pseudo_type = "US"
        relativistic = attributes.get("relativistic", "")
        has_so = attributes.get("has_so", "").lower() in ("t", "true", ".true.")
    else:
        lines = [line.strip() for line in (match[2] or "").splitlines() if line.strip()]
        if len(lines) < 6:
            raise ValueError("Incomplete UPF 1 header.")
        actual = lines[1].split()[0]
        functional = lines[4]
        pseudo_type = lines[2].split()[0].upper()
        has_so = "<PP_SPIN_ORB" in text or "<PP_ADDINFO" in text
        relativistic = "full" if has_so else "scalar"
    if actual != element:
        raise ValueError(f"UPF element {actual!r} does not match {element}.")
    return {"element": actual, "functional": functional, "pseudo_type": pseudo_type, "relativistic": relativistic, "has_so": has_so, "sha256": hashlib.sha256(data).hexdigest()}


def ensure_pseudos(config, elements, transport=None, *, cache_dir=None, soc=False, library="sssp") -> dict:
    """Install only selected public UPFs, never replacing a different existing file."""
    plan = plan_pseudopotentials(elements, soc=soc, library=library)
    destination = _absolute(getattr(config, "qe_pseudo_root", ""))
    cache = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".dft-agent" / "downloads" / "qe"
    if library == "pseudodojo":
        archive_metadata = {"url": plan["source_url"], "sha256": plan["archive_sha256"]}
        license_metadata = plan["license"]
        archive = _download("nc-sr-04_pbe_standard_upf.tgz", archive_metadata, cache)
        license_path = _download("PseudoDojo-README.md", license_metadata, cache)
    else:
        archive = _download("SSSP_1.3.0_PBE_efficiency.tar.gz", PSEUDO_ARCHIVE, cache)
        license_metadata = PSEUDO_LICENSE
        license_path = _download("SSSP-LICENSE.txt", license_metadata, cache)
    with tempfile.TemporaryDirectory(prefix="dft-qe-pseudos-") as folder, tarfile.open(archive, "r:gz") as source, _connection(config, transport) as active:
        folder = Path(folder)
        members = {}
        for member in source:
            name = PurePosixPath(member.name).name
            if name in {v.get("archive_filename", v["filename"]) for v in plan["elements"].values()}:
                if not member.isfile() or name in members:
                    raise ValueError("Invalid or duplicate UPF archive member.")
                members[name] = member
        items = []
        for element, metadata in plan["elements"].items():
            name = metadata["filename"]
            archive_name = metadata.get("archive_filename", name)
            if archive_name not in members:
                raise ValueError(f"{plan['library']} archive is missing {archive_name}.")
            with source.extractfile(members[archive_name]) as handle:
                data = handle.read(50 * 1024 * 1024 + 1)
            header = validate_upf(data, element)
            if header["sha256"] != metadata["sha256"]:
                raise ValueError(f"UPF SHA-256 mismatch for {element}.")
            if library == "pseudodojo" and (header["pseudo_type"] != "NC" or header["functional"] != "PBE" or header["has_so"]):
                raise ValueError(f"{element} is not a scalar-relativistic PBE NC pseudopotential.")
            metadata["header"] = header
            path = folder / name
            path.write_bytes(data)
            items.append((path, metadata["sha256"]))
        manifest = folder / (library + "-pbe.json")
        manifest.write_text(json.dumps(plan, indent=2) + "\n")
        # A separate manifest for each selection keeps earlier records intact.
        manifest = manifest.rename(folder / (library + "-" + _digest(manifest)[:12] + ".json"))
        items.extend([(license_path, license_metadata["sha256"]), (manifest, _digest(manifest))])
        remote_archive = None
        if library == "pseudodojo":
            remote_sources = destination + "/.pseudodojo-0.4-source"
            transfers = _source_transfers(active, {archive.name: archive_metadata}, remote_sources, cache)
            if not transfers:
                remote_archive = remote_sources + "/" + archive.name
        for path, digest in items:
            target = destination + "/" + path.name
            code = f'''import hashlib,json,pathlib
p=pathlib.Path({target!r})
exists=p.exists() or p.is_symlink()
same=exists and p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()=={digest!r}
print({_MARKER!r}+json.dumps({{'exists':exists,'same':same}}))
'''
            existing = _json_run(active, code)
            if existing["exists"] and not existing["same"]:
                raise ValueError(f"{target} already contains a different file; choose another UPF folder.")
            if not existing["same"]:
                staged = target + "." + digest[:12] + ".part"
                selected = next((v for v in plan["elements"].values() if v["filename"] == path.name), None)
                if remote_archive and selected:
                    code = f'''import hashlib,json,pathlib,tarfile
with tarfile.open({remote_archive!r}, 'r:gz') as archive:
    members=[m for m in archive.getmembers() if pathlib.PurePosixPath(m.name).name=={selected['archive_filename']!r}]
    if len(members)!=1 or not members[0].isfile(): raise RuntimeError('Invalid UPF archive member')
    data=archive.extractfile(members[0]).read(50*1024*1024+1)
if hashlib.sha256(data).hexdigest()!={digest!r}: raise RuntimeError('UPF archive checksum mismatch')
path=pathlib.Path({staged!r})
if path.is_symlink(): raise RuntimeError('Unsafe UPF staging path')
path.write_bytes(data)
print({_MARKER!r}+json.dumps({{'staged':str(path)}}))
'''
                    _json_run(active, code)
                else:
                    active.upload(path, staged)
                code = f'''import hashlib,json,os,pathlib
source=pathlib.Path({staged!r}); target=pathlib.Path({target!r})
if hashlib.sha256(source.read_bytes()).hexdigest()!={digest!r}: raise RuntimeError('UPF transfer checksum mismatch')
try: os.link(str(source),str(target))
except FileExistsError:
    if target.is_symlink() or not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!={digest!r}: raise RuntimeError('Existing UPF changed; refusing replacement')
source.unlink()
print({_MARKER!r}+json.dumps({{'installed':str(target)}}))
'''
                _json_run(active, code)
    mapping = {element: {"filename": value["filename"], "sha256": value["sha256"],
                        "ecutwfc": value.get("cutoff_wfc", 0), "ecutrho": value.get("cutoff_rho", 0),
                        "element": element, "functional": value["header"]["functional"],
                        "pseudo_type": value["header"]["pseudo_type"],
                        "relativistic": value["header"]["relativistic"],
                        "fully_relativistic": value["header"]["has_so"]}
               for element, value in plan["elements"].items()}
    return dict(plan, pseudo_root=destination, pseudopotentials={element: value["filename"] for element, value in plan["elements"].items()}, config_pseudopotentials=mapping)
