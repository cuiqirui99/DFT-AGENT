"""Durable Quantum ESPRESSO jobs with verified inputs and checkpoints."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import tempfile
import uuid
import zipfile

from .config import ClusterConfig
from .transport import SSHTransport, TransportError
from . import qe
from .workflow import (_lock, _save, _write, _digest, _command, _poll,
                       _check_config, _config_attachment_lock,
                       _config_attachment_workspace, _commit_config_attachment,
                       read_state, utc_now, TERMINAL, SLURM_TERMINAL, MAX_REMOTE_FAILURES)

PROGRAMS = {"relax": "pw.x", "vc-relax": "pw.x", "scf": "pw.x",
            "nscf": "pw.x", "bands": "pw.x", "dos": "dos.x",
            "projwfc": "projwfc.x", "ph": "ph.x", "q2r": "q2r.x",
            "matdyn": "matdyn.x", "neb": "neb.x", "pp": "pp.x"}
PROGRAMS.update(eph_dense_scf="pw.x", eph_scf="pw.x", eph_ph="ph.x",
                eph_q2r="q2r.x", eph_matdyn="matdyn.x", eph_lambda="lambda.x")
PROGRAMS.update(epw_iso="epw.x", epw_tc="epw.x", epw_aniso="epw.x")
PROGRAMS.update(wannier_scf="pw.x", wannier_ph="ph.x", wannier_nscf="pw.x", epw_interpolate="epw.x")
ALIASES = {"pdos": "projwfc", "phonon": "ph", "phonons": "ph",
           "phonon_bands": "phonon_bands", "phonon-bands": "phonon_bands", "charge_density": "pp",
           "electron-phonon": "electron_phonon"}
RECEIPTS = {"execution.json", "input_hashes.sha256", "pseudo_manifest.json",
            "checkpoint.json", "runtime.json", "slurm.out", "slurm.err",
            "data-file-schema.xml", "dft_agent.a2Fsave"}
_RESULT_NAME = re.compile(r"(?:pdos[^/]*|dft_agent\.(?:dyn[0-9]*(?:\.xml)?|dat|int|path|axsf|xyz)|phonon[^/]*|force_constants(?:\.xml)?|dos\.dat|charge_density\.(?:dat|cube)|a2F\.dos[0-9]+|lambda(?:\.dat)?|alpha2F\.dat|elph_dir/[A-Za-z0-9_.+-]+)")
_EPW_RESULT_NAME = re.compile(r"(?:dft_agent\.(?:wout|win|a2f|imag_aniso_[0-9.]+)|crystal\.fmt|selecq\.fmt|scratch/dft_agent\.ephmat/(?:freq|egnv|ikmap|npool|ephmat[0-9]*))")

_QE_UPLOAD_NAMES = frozenset({"pw.in", "ph.in", "q2r.in", "matdyn.in", "dos.in", "projwfc.in",
    "neb.in", "pp.in", "lambda.in", "structure.cif", "qe-input.json", "pseudo_manifest.json",
    "runtime.json", "qe_runtime.py", "submit.sh", "dispatch.py",
    "epw.in", "alpha2f.dat", "source.a2f", "source-schema.xml", "source-epw.in", "source-epw.out", "source-wannier.wout", "crystal.fmt"})


def _require_cpu(config):
    if config is not None and config.compute_backend != "cpu":
        raise ValueError("QE GPU execution has not been verified. Select CPU for this version.")


def _science(task):
    if task.startswith("wannier_") or task == "epw_interpolate":
        from . import qe_wannier
        return qe_wannier
    if task.startswith("epw_"):
        from . import qe_epw
        return qe_epw
    if task.startswith("eph_"):
        from . import qe_eph
        return qe_eph
    return qe


def _artifact_name(value):
    if isinstance(value, str) and (re.fullmatch(r"elph_dir/[A-Za-z0-9_.+-]+", value) or _EPW_RESULT_NAME.fullmatch(value)):
        return value
    return _name(value)


def _name(value):
    if not isinstance(value, str) or not value or PurePosixPath(value).name != value or value in {".", ".."} or any(c in value for c in "\r\n\x00"):
        raise ValueError("QE file names must be plain file names.")
    return value


def _epw_ranks(tokens, binary_index, tasks):
    """Resolve the rank count before fixing EPW's one-rank-per-pool layout."""
    if binary_index == 0:
        ranks = 1
    else:
        launcher = PurePosixPath(tokens[0]).name
        if launcher not in {"srun", "mpirun", "mpiexec"} or ":" in tokens:
            raise ValueError("EPW needs a direct launch, srun, mpirun or mpiexec with a fixed rank count.")
        options = {"-n", "--ntasks"} if launcher == "srun" else {"-n", "-np", "--np", "--n"}
        values, i = [], 1
        while i < binary_index:
            token = tokens[i]
            value = None
            if token in options:
                i += 1
                if i >= binary_index:
                    raise ValueError("The MPI rank count is missing.")
                value = tokens[i]
            else:
                for option in sorted(options, key=len, reverse=True):
                    if token.startswith(option + "="):
                        value = token[len(option) + 1:]
                        break
                    if option in {"-n", "-np"} and re.fullmatch(re.escape(option) + r"\d+", token):
                        value = token[len(option):]
                        break
            if value is not None:
                if not re.fullmatch(r"[1-9][0-9]*", value):
                    raise ValueError("Use a positive integer for the MPI rank count.")
                values.append(int(value))
            i += 1
        if len(values) > 1:
            raise ValueError("Specify the MPI rank count once.")
        if not values and launcher != "srun":
            raise ValueError("For EPW, set mpirun or mpiexec -np to the configured task count.")
        ranks = values[0] if values else tasks
    if ranks != tasks:
        raise ValueError("The MPI rank count must match the configured task count for EPW.")
    return ranks


def _epw_pools(tokens, binary_index, ranks):
    # QE 7.4.1 EPW requires npool * nimage == nproc. These adapters use one image.
    result, i = tokens[:binary_index + 1], binary_index + 1
    pools = {"-nk", "-npool", "-npools"}
    images = {"-ni", "-nimage", "-nimages", "-npot"}
    while i < len(tokens):
        option = tokens[i]
        if option in pools | images:
            if i + 1 >= len(tokens) or not re.fullmatch(r"[1-9][0-9]*", tokens[i + 1]):
                raise ValueError("QE pool and image counts must be positive integers.")
            if option in images and int(tokens[i + 1]) != 1:
                raise ValueError("EPW workflows use one image. Remove the image-splitting option.")
            i += 2
        else:
            result.append(option)
            i += 1
    return result + ["-npool", str(ranks)]


def _command_for(config, program):
    if program not in PROGRAMS.values():
        raise ValueError("Unsupported QE program.")
    tokens = shlex.split(config.qe_command)
    matches = [i for i, token in enumerate(tokens) if PurePosixPath(token).name == "pw.x"]
    if len(matches) != 1 or any(token in {";", "&&", "||", "|", ">", "<"} for token in tokens):
        raise ValueError("QE command must launch pw.x, optionally through srun or mpirun.")
    index = matches[0]
    binary = tokens[index]
    directory = config.qe_bin_dir or (str(PurePosixPath(binary).parent) if "/" in binary else "")
    tokens[index] = str(PurePosixPath(directory) / program) if directory else program
    # Input and output files are controlled by the frozen stage.
    if any(token in {"-in", "-inp", "-input", "--input"} for token in tokens):
        raise ValueError("Leave input-file options out of the QE launch command.")
    if program == "epw.x":
        ranks = _epw_ranks(tokens, index, config.tasks)
        tokens = _epw_pools(tokens, index, ranks)
    return shlex.join(tokens)


def _sequence(tasks, parameters, stage_parameters):
    if not isinstance(tasks, (list, tuple)) or not tasks:
        raise ValueError("Choose at least one QE task.")
    if stage_parameters is not None and (len(stage_parameters) != len(tasks) or any(not isinstance(p, dict) for p in stage_parameters)):
        raise ValueError("Supply one parameter override per task.")
    sequence, relaxed, scf, nscf, phonon, force, coupling, interpolation = [], None, None, None, None, None, None, None

    def add(name, recipe, restart=None):
        sequence.append({"name": name, "parameters": deepcopy(recipe),
                         "structure_from": relaxed, "restart_from": restart, "request_index": index})
        return len(sequence) - 1

    def charge(recipe):
        nonlocal scf, nscf
        # Different stage recipes get their own SCF rather than borrowing a
        # charge density from another functional, spin state or k-point mesh.
        if scf is None or sequence[scf]["parameters"] != recipe:
            scf = add("scf", recipe)
            nscf = None
        return scf

    def phonon_postprocess(recipe, parent):
        inherited = deepcopy(sequence[parent]["parameters"])
        for key, value in recipe.items():
            if key == "namelists":
                native = inherited.setdefault("namelists", {})
                if (not isinstance(native, dict) or not isinstance(value, dict)
                        or any(not isinstance(settings, dict) for settings in [*native.values(), *value.values()])):
                    raise ValueError("namelists must map section names to typed options.")
                for section, settings in value.items():
                    original = native.setdefault(section, {})
                    for name, setting in settings.items():
                        if section not in {"q2r", "matdyn"} and original.get(name) != setting:
                            raise ValueError("Postprocessing cannot change the source phonon settings. Prepare a new phonon calculation.")
                        original[name] = deepcopy(setting)
            elif key in {"asr", "qpoints", "line_density"}:
                inherited[key] = deepcopy(value)
            elif key not in inherited or value != inherited[key]:
                raise ValueError("Postprocessing cannot change the source phonon settings. Prepare a new phonon calculation.")
        return inherited

    for index, requested in enumerate(tasks):
        task = ALIASES.get(requested, requested)
        if task not in {*PROGRAMS, "phonon_bands", "electron_phonon", "wannier_electron_phonon"}:
            raise ValueError(f"Unsupported QE task: {requested}")
        recipe = {**deepcopy(parameters), **deepcopy((stage_parameters or [{}] * len(tasks))[index])}
        if "q_mesh" in recipe or "phonon_grid" in recipe:
            recipe["qmesh"] = recipe.pop("q_mesh", recipe.pop("phonon_grid", None))
        if task == "wannier_electron_phonon":
            parent = add("wannier_scf", recipe)
            derivatives = add("wannier_ph", recipe, parent)
            wavefunctions = add("wannier_nscf", recipe, parent)
            interpolation = add("epw_interpolate", recipe, wavefunctions)
            sequence[interpolation]["auxiliary_from"] = derivatives
        elif task.startswith("wannier_") or task == "epw_interpolate":
            raise ValueError("Request wannier_electron_phonon to prepare its SCF, phonon and Wannier stages.")
        elif task == "electron_phonon":
            previous = None
            chain = ["eph_dense_scf", "eph_scf", "eph_ph"]
            if recipe.get("eph_integration", "interpolated") == "interpolated":
                chain += ["eph_q2r", "eph_matdyn"]
            for name in [*chain, "eph_lambda"]:
                previous = add(name, recipe, previous)
            coupling = previous
        elif task == "epw_aniso":
            if interpolation is None:
                raise ValueError("Anisotropic EPW needs accepted Wannier interpolation. Add wannier_electron_phonon first.")
            current = add(task, recipe, interpolation)
            sequence[current]["evidence_from"] = interpolation
        elif task.startswith("epw_"):
            sources = [item for item in (coupling, interpolation) if item is not None]
            if not sources:
                raise ValueError("EPW needs an accepted electron-phonon spectrum. Add electron_phonon or wannier_electron_phonon first, or continue an accepted spectrum.")
            current = add(task, recipe)
            sequence[current]["evidence_from"] = max(sources)
        elif task.startswith("eph_"):
            raise ValueError("Request electron_phonon to prepare the complete validated dependency chain.")
        elif task in {"relax", "vc-relax"}:
            if sequence:
                raise ValueError("Put relaxation first in the QE plan.")
            relaxed = add(task, recipe)
            scf = nscf = None
        elif task == "scf":
            scf, nscf = add(task, recipe), None
        elif task in {"bands", "nscf"}:
            parent = charge(recipe)
            current = add(task, recipe, parent)
            if task == "nscf":
                nscf = current
        elif task in {"dos", "projwfc"}:
            parent = charge(recipe)
            if nscf is None or sequence[nscf]["parameters"] != recipe:
                nscf = add("nscf", recipe, parent)
            add(task, recipe, nscf)
        elif task in {"ph", "phonon_bands"}:
            parent = charge(recipe)
            ph_recipe = deepcopy(recipe)
            if task == "phonon_bands":
                grid = ph_recipe.get("qmesh", [2, 2, 2])
                if not isinstance(grid, (list, tuple)) or len(grid) != 3 or any(type(n) is not int or n < 1 for n in grid):
                    raise ValueError("Use three positive integers for q_mesh.")
                ph_recipe["qmesh"] = list(grid)
                ph_recipe.setdefault("namelists", {}).setdefault("inputph", {}).update(
                    ldisp=True, nq1=grid[0], nq2=grid[1], nq3=grid[2])
            phonon = add("ph", ph_recipe, parent)
            if task == "phonon_bands":
                force = add("q2r", recipe, phonon)
                add("matdyn", recipe, force)
        elif task == "q2r":
            if phonon is None:
                raise ValueError("q2r requires a preceding phonon grid.")
            force = add(task, phonon_postprocess(recipe, phonon), phonon)
        elif task == "matdyn":
            if force is None:
                raise ValueError("matdyn requires preceding q2r force constants.")
            add(task, phonon_postprocess(recipe, force), force)
        elif task == "pp":
            add(task, recipe, charge(recipe))
        else:
            if len(tasks) != 1:
                raise ValueError("Prepare NEB as a separate task with its own images.")
            add(task, recipe)
    return sequence


def _freeze_images(root, recipe, sources):
    paths = recipe.get("images")
    if not paths:
        return
    frozen = []
    for index, value in enumerate(paths):
        source = Path(value).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        relative = f"source/neb-{index:03d}{source.suffix or '.vasp'}"
        target = root / relative
        if target.exists() and _digest(target) != _digest(source):
            raise ValueError("NEB image sets differ between stages.")
        shutil.copyfile(source, target)
        sources[relative] = _digest(target)
        frozen.append(relative)
    recipe["images"] = frozen


def prepare_plan(structure_path, run_dir, config, tasks, parameters=None,
                 magnetic_states=None, initial_moment=3.0, stage_parameters=None, **kwargs):
    """Freeze a QE plan locally. Only advance() can submit it."""
    if magnetic_states:
        raise ValueError("Prepare each QE magnetic state as a separate reviewed plan.")
    if config is not None and config.engine != "qe":
        raise ValueError("Select Quantum ESPRESSO in cluster settings.")
    _require_cpu(config)
    source = Path(structure_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    root = Path(run_dir).expanduser().resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise FileExistsError("This run already exists. Resume it or choose a new folder.")
    parameters = deepcopy(parameters or {})
    sequence = _sequence(tasks, parameters, stage_parameters)
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".qe-plan-", dir=root.parent))
    try:
        (temporary / "source").mkdir()
        relative = "source/structure" + (source.suffix or ".vasp")
        shutil.copyfile(source, temporary / relative)
        sources = {relative: _digest(temporary / relative)}
        run_id = "qe-" + uuid.uuid4().hex[:16]
        stages = []
        for index, spec in enumerate(sequence):
            _freeze_images(temporary, spec["parameters"], sources)
            stage = {**spec, "source_path": relative, "folder": f"{index + 1:02d}_{spec['name']}",
                     "job_name": f"{run_id}-{index + 1:02d}-{spec['name']}", "job_id": None,
                     "status": "planned", "metadata": {}, "staged": False,
                     "materialized": False, "result": None}
            stages.append(stage)
        plan = {"schema_version": 1, "engine": "qe", "tasks": list(tasks), "parameters": deepcopy(parameters), "sources": sources,
                "stages": [{k: deepcopy(stage[k]) for k in
                            ("name", "parameters", "source_path", "folder", "structure_from", "restart_from", "request_index")}
                           for stage in stages]}
        for spec, stage in zip(plan["stages"], stages):
            for key in ("evidence_from", "auxiliary_from"):
                if key in stage:
                    spec[key] = stage[key]
        if stage_parameters is not None:
            plan["stage_parameters"] = deepcopy(stage_parameters)
        _write(temporary / "plan.json", plan)
        if config:
            config.save(temporary / "config.json")
        state = {"schema_version": 2, "engine": "qe", "run_id": run_id,
                 "task": tasks[0] if len(tasks) == 1 else "pipeline", "tasks": list(tasks),
                 "parameters": parameters, "formula": "", "status": "planned", "current_stage": 0,
                 "stages": stages, "history": [], "created_at": utc_now(), "last_error": None,
                 "cancel_requested": False, "plan_sha256": _digest(temporary / "plan.json"),
                 "config_sha256": _digest(temporary / "config.json") if config else None,
                 "remote_root": config.remote_root.rstrip("/") + "/" + run_id if config else None}
        # Check every recipe before any remote work. Dependent inputs are rebuilt
        # from the accepted structure, not this provisional geometry.
        for stage in stages:
            _materialize(temporary, state, config, stage, preview=stage["structure_from"] is not None)
        state["formula"] = stages[0]["metadata"].get("formula", "")
        _save(temporary, state, "QE plan prepared. Ready to submit.")
        os.replace(temporary, root)
        return state
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare_run(structure_path, run_dir, config, task="relax", parameters=None):
    return prepare_plan(structure_path, run_dir, config, [task], parameters)


def _materialize(root, state, config, stage, preview=False):
    if stage["materialized"]:
        return
    source = root / stage["source_path"]
    if stage["structure_from"] is not None and not preview:
        parent = state["stages"][stage["structure_from"]]
        result = parent.get("result") or {}
        if parent["status"] != "succeeded" or not result.get("success"):
            raise ValueError("The relaxed structure has not been accepted.")
        native = result.get("final_native_structure_sha256")
        source = root / parent["folder"] / "outputs" / ("CONTCAR" if native else "final.cif")
        if not source.is_file() or _digest(source) != (native or result.get("final_structure_sha256")):
            raise ValueError("The accepted relaxed structure changed or is missing.")
    recipe = deepcopy(stage["parameters"])
    if recipe.get("images"):
        recipe["images"] = [str(root / name) for name in recipe["images"]]
    evidence_dir = None
    if (stage["name"] == "eph_lambda" or stage.get("evidence_from") is not None) and not stage.get("source_evidence"):
        parent_index = stage["restart_from"] if stage["name"] == "eph_lambda" else stage["evidence_from"]
        parent = state["stages"][parent_index]
        if parent["status"] != "succeeded" or not (parent.get("result") or {}).get("success"):
            preview = True
        elif not preview:
            evidence_dir = root / parent["folder"] / "outputs"
            records = json.loads((evidence_dir / "artifact_manifest.json").read_text())
            if any(_digest(evidence_dir / _artifact_name(name)) != value["sha256"] for name, value in records.items()):
                raise ValueError("Accepted electron-phonon evidence changed.")
    if stage.get("source_evidence"):
        evidence_dir = root / stage["source_evidence"]
        plan = json.loads((root / "plan.json").read_text())
        records = {name: digest for name, digest in plan["sources"].items()
                   if name.startswith(stage["source_evidence"] + "/")}
        if (not records or not evidence_dir.resolve().is_relative_to(root.resolve())
                or any(_digest(root / name) != digest for name, digest in records.items())):
            raise ValueError("The accepted electron-phonon source changed.")
    inputs = root / stage["folder"] / ("preview" if preview else "inputs")
    if inputs.exists():
        shutil.rmtree(inputs)
    pseudos = config.qe_pseudopotentials if config else recipe.get("pseudopotentials", {})
    extra = {"evidence_dir": evidence_dir} if stage["name"].startswith(("eph_", "epw_")) else {}
    metadata = _science(stage["name"]).prepare_inputs(source, inputs, stage["name"], recipe, pseudos, **extra)
    for key in ("input_file", "output_file"):
        _name(metadata[key])
    names = list(dict.fromkeys([*metadata.get("input_files", [metadata["input_file"], "structure.cif"]), "qe-input.json"]))
    metadata["input_sha256"] = {_name(name): _digest(inputs / name) for name in names}
    if config:
        _command_for(config, PROGRAMS[stage["name"]])
    if preview:
        shutil.rmtree(inputs)
        return
    stage["metadata"] = metadata
    stage["materialized"] = True
    _write(inputs / "metadata.json", metadata)


def attach_config(run_dir, config):
    if config is None or config.engine != "qe":
        raise ValueError("Save Quantum ESPRESSO cluster settings first.")
    _require_cpu(config)
    root = Path(run_dir).expanduser().resolve()
    with _config_attachment_lock(root):
        original = read_state(root)
        _check_config(root, original)
        if original["status"] != "planned" or any(s.get("staged") or s.get("job_id") for s in original["stages"]):
            raise ValueError("Cluster settings can only be attached before submission.")
        with _config_attachment_workspace(root) as temporary:
            prepared = temporary / "run"
            shutil.copytree(root, prepared, ignore=shutil.ignore_patterns(".state.lock", ".worker.lock"))
            state = read_state(prepared)
            config.save(prepared / "config.json")
            state.update(config_sha256=_digest(prepared / "config.json"),
                         remote_root=config.remote_root.rstrip("/") + "/" + state["run_id"])
            for stage in state["stages"]:
                stage["materialized"] = False
                stage["metadata"] = {}
                _materialize(prepared, state, config, stage, preview=stage["structure_from"] is not None)
            _save(prepared, state, "QE cluster settings attached.")
            _commit_config_attachment(root, prepared,
                ["config.json", *(s["folder"] + "/inputs" for s in state["stages"]), "run.json"],
                temporary / "original")
            return state


# This small stdlib program runs on the cluster. Checkpoints stay there; only
# hashes, small scientific outputs and requested raw files cross SSH.
_RUNTIME = r'''import datetime, hashlib, json, os, shutil, sys
from pathlib import Path

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def safe(root, name):
    path = root / name
    if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
        raise RuntimeError('Unsafe checkpoint path')
    if root.resolve() not in path.resolve().parents:
        raise RuntimeError('Checkpoint escaped its stage')
    return path

def checkpoint_name(name):
    return name.startswith(('scratch/', 'elph_dir/', 'a2F.dos')) or Path(name).name.startswith('dft_agent.dyn') or name in {'force_constants', 'force_constants.xml', 'lambda', 'lambda.dat', 'alpha2F.dat', 'phonon.dos', 'crystal.fmt', 'selecq.fmt', 'dft_agent.a2f'}

root = Path.cwd()
spec = json.loads((root/'runtime.json').read_text())
if sys.argv[1] == 'prepare':
    for name, expected in spec['input_sha256'].items():
        if sha(safe(root, name)) != expected: raise RuntimeError('Prepared input changed: '+name)
    (root/'.execution_started').open('x').close()
    (root/'input_hashes.sha256').write_text(''.join(h+'  '+n+'\n' for n,h in spec['input_sha256'].items()))
    for dep in spec['dependencies']:
        source = Path(dep['directory'])
        receipt = source/'checkpoint.json'
        if sha(receipt) != dep['sha256']: raise RuntimeError('Accepted checkpoint receipt changed')
        records = json.loads(receipt.read_text())
        mapping = dep.get('mapping')
        if mapping is None:
            mapping = {name: name for name in records}
        elif not isinstance(mapping, dict) or not mapping or any(
                not destination.startswith('save/') for destination in mapping.values()):
            raise RuntimeError('Invalid auxiliary phonon checkpoint mapping')
        if len(set(mapping.values())) != len(mapping):
            raise RuntimeError('Duplicate checkpoint destinations')
        for name, destination in mapping.items():
            if name not in records: raise RuntimeError('Checkpoint source is missing')
            value = records[name]
            if not checkpoint_name(name): raise RuntimeError('Unexpected checkpoint member')
            old = safe(source, name)
            if old.stat().st_size != value['size'] or sha(old) != value['sha256']:
                raise RuntimeError('Accepted checkpoint changed: '+name)
            new = safe(root, destination)
            if new.exists(): raise RuntimeError('Checkpoint would overwrite a prepared or inherited file')
            new.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(old, new)
    (root/'pseudo').mkdir(exist_ok=True)
    for name, value in spec['pseudos'].items():
        old = safe(Path(spec['pseudo_root']), name)
        if old.stat().st_size != value['size'] or sha(old) != value['sha256']:
            raise RuntimeError('Pseudopotential changed: '+name)
        shutil.copyfile(old, root/'pseudo'/name)
    (root/'execution.json').write_text(json.dumps(dict(job_id=os.environ.get('SLURM_JOB_ID'),
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        runtime_sha256=sha(root/'runtime.json'), engine='qe')))
else:
    code = int(sys.argv[2])
    receipt = json.loads((root/'execution.json').read_text())
    receipt.update(exit_code=code, finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (root/'execution.json').write_text(json.dumps(receipt))
    xml = root/'scratch'/'dft_agent.save'/'data-file-schema.xml'
    if xml.is_file(): shutil.copyfile(xml, root/'data-file-schema.xml')
    dense = root/'scratch'/'dft_agent.a2Fsave'
    if dense.is_file(): shutil.copyfile(dense, root/'dft_agent.a2Fsave')
    paths = list((root/'scratch').rglob('*')) if (root/'scratch').is_dir() else []
    if (root/'elph_dir').is_dir(): paths += list((root/'elph_dir').rglob('*'))
    paths += [p for p in root.iterdir() if checkpoint_name(p.name)]
    manifest = {}
    for path in paths:
        name = path.relative_to(root).as_posix()
        if path.is_file() and not path.is_symlink():
            path = safe(root, name)
            manifest[name] = dict(size=path.stat().st_size, sha256=sha(path))
    (root/'checkpoint.json').write_text(json.dumps(manifest, sort_keys=True))
'''


# Embedded for SSH and frozen desktop builds; no Python source lookup at runtime.
_UPF_VALIDATOR = r'''def validate_upf(data: bytes, element: str) -> dict:
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
'''


def _pseudo_snapshot(transport, config, stage, inherited=None):

    values = stage["metadata"].get("pseudo_files", [])
    if isinstance(values, dict):
        values = list(values.values())
    names = sorted({_name(value["filename"] if isinstance(value, dict) else value) for value in values} or set(inherited or {}))
    if not names:
        return {}
    entries = {}
    for element, value in config.qe_pseudopotentials.items():
        entry = dict(value) if isinstance(value, dict) else {"filename": value}
        name = entry.get("filename")
        if name not in names:
            continue
        if name in entries and entries[name]["element"] != element:
            raise ValueError("One UPF file cannot represent different elements.")
        entries[name] = {**entry, "element": element}
    if set(entries) != set(names):
        raise ValueError("The prepared UPFs do not match the cluster library mapping.")
    system = stage["metadata"].get("namelists", {}).get("system", {})
    require_soc = bool(stage.get("parameters", {}).get("soc") or system.get("lspinorb"))
    # Send the same small header validator used for library imports, never UPF
    # contents. The returned receipt binds the checked header to the actual file.
    code = "import hashlib,json,re\nfrom pathlib import Path\n" + _UPF_VALIDATOR
    code += """
root=Path(%r)
result={}
for name,expected in %r.items():
    path=root/name
    if not path.is_file() or path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise RuntimeError('Missing or unsafe pseudopotential: '+name)
    if path.stat().st_size > 50*1024*1024:
        raise RuntimeError('Pseudopotential is unexpectedly large: '+name)
    data=path.read_bytes()
    header=validate_upf(data,expected['element'])
    if %r and not header['has_so']:
        raise RuntimeError('SOC requires spin-orbit data in the actual UPF: '+name)
    result[name]={'sha256':header['sha256'],'size':len(data),'element':header['element'],
                  'fully_relativistic':header['has_so'],'functional':header['functional'],
                  'pseudo_type':header['pseudo_type']}
print(json.dumps(result))
""" % (config.qe_pseudo_root, entries, require_soc)
    result = json.loads(_command(transport, "python3 -c " + shlex.quote(code)))
    if (set(result) != set(names) or any(not isinstance(value, dict)
            or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))
            or type(value.get("size")) is not int or value["size"] <= 0
            or value.get("element") != entries[name]["element"]
            or type(value.get("fully_relativistic")) is not bool
            or (require_soc and not value["fully_relativistic"])
            for name, value in result.items())):
        raise ValueError("Invalid pseudopotential receipt.")
    for name, entry in entries.items():
        if entry.get("sha256") and result[name]["sha256"] != entry["sha256"]:
            raise ValueError("A QE pseudopotential differs from its validated library record.")
    if stage.get("name", "").startswith("wannier_") or stage.get("name") == "epw_interpolate":
        from .qe_wannier import _require_nc
        _require_nc(result)
    effective = deepcopy(stage.get("parameters", {}))
    # Check generated defaults as well as explicit requests, including older
    # frozen inputs that still enabled unsupported noncollinear hybrid stress.
    native = stage["metadata"].get("namelists", {})
    for section, keys in {"control": ("tstress", "tprnfor"), "system": ("input_dft", "noncolin", "lspinorb")}.items():
        values = {key: native[section][key] for key in keys if key in native.get(section, {})}
        if values:
            effective.setdefault("namelists", {}).setdefault(section, {}).update(values)
    qe.validate_compatibility(stage.get("name", stage["metadata"].get("canonical_task", "scf")),
                              effective, result, verified=True)
    return result


def _script(config, stage):
    metadata = stage["metadata"]
    launch = _command_for(config, PROGRAMS[stage["name"]])
    if metadata.get("no_mpi"):
        tokens = shlex.split(launch)
        launch = shlex.quote(next(token for token in tokens if PurePosixPath(token).name == PROGRAMS[stage["name"]]))
    launch += (" < " if stage["name"] == "eph_lambda" else " -in ") + shlex.quote(metadata["input_file"])
    launch += " > " + shlex.quote(metadata["output_file"])
    lines = ["#!/bin/bash", f"#SBATCH --partition={config.partition}",
             f"#SBATCH --nodes={config.nodes}", f"#SBATCH --ntasks={config.tasks}",
             f"#SBATCH --time={config.walltime}", "#SBATCH --output=slurm.out", "#SBATCH --error=slurm.err"]
    if config.account:
        lines.append(f"#SBATCH --account={config.account}")
    lines.extend(config.sbatch_resource_lines())
    lines.extend(["set -e", *config.setup_commands, "python3 qe_runtime.py prepare", "set +e",
                  launch,
                  "code=$?", "set -e", 'python3 qe_runtime.py finish "$code"', 'exit "$code"'])
    return "\n".join(lines) + "\n"


def _validate_checkpoint(records, task):
    if not isinstance(records, dict) or not records:
        raise ValueError("The required QE checkpoint is empty.")
    for name, value in records.items():
        relative = PurePosixPath(name)
        if (relative.is_absolute() or ".." in relative.parts or not isinstance(value, dict)
                or type(value.get("size")) is not int or value["size"] < 0
                or not re.fullmatch(r"[0-9a-f]{64}", value.get("sha256", ""))):
            raise ValueError("The QE checkpoint receipt is invalid.")
    names = {name for name, value in records.items() if value["size"] > 0}
    if task in {"scf", "relax", "vc-relax", "nscf", "bands", "ph", "projwfc", "pp", "dos", "eph_scf", "eph_ph", "wannier_ph", "wannier_nscf", "epw_interpolate"}:
        if "scratch/dft_agent.save/data-file-schema.xml" not in names or not any(
                name.startswith("scratch/dft_agent.save/charge-density.") for name in names):
            raise ValueError("The required QE charge density or XML checkpoint is missing.")
    if task in {"scf", "relax", "vc-relax", "ph", "projwfc", "eph_ph", "wannier_ph", "epw_interpolate"} and not any(
            name.startswith("scratch/") and re.search(r"(?:^|[._/])wfc(?:up|dw)?[0-9]*(?:[._]|$)", name) for name in names):
        raise ValueError("The required QE wavefunction checkpoint is missing.")
    if task in {"eph_scf", "eph_ph"} and "scratch/dft_agent.a2Fsave" not in names:
        raise ValueError("The dense electronic-grid checkpoint is missing.")
    if task in {"q2r", "eph_q2r"} and ("dft_agent.dyn0" not in names or "dft_agent.dyn1" not in names and "dft_agent.dyn1.xml" not in names):
        raise ValueError("The phonon-grid dynamical matrices are missing.")
    if task in {"matdyn", "eph_matdyn"} and not names.intersection({"force_constants", "force_constants.xml"}):
        raise ValueError("The interpolated force-constant checkpoint is missing.")
    if task == "eph_lambda" and ("dft_agent.dyn0" not in names or "elph_dir/elph.inp_lambda.1" not in names):
        raise ValueError("The electron-phonon checkpoint is missing.")
    if task == "epw_aniso":
        required = {"crystal.fmt", "selecq.fmt", "dft_agent.a2f"}
        required.update("scratch/dft_agent.ephmat/" + name for name in ("freq", "egnv", "ikmap", "npool"))
        if not required.issubset(names) or not any(re.fullmatch(r"scratch/dft_agent\.ephmat/ephmat[0-9]*", name) for name in names):
            raise ValueError("The anisotropic electron-phonon checkpoint is incomplete.")


def _submit(root, state, config, transport):
    _require_cpu(config)
    stage = state["stages"][state["current_stage"]]
    _materialize(root, state, config, stage)
    if not stage["materialized"] or stage.get("metadata", {}).get("preview_only"):
        raise ValueError("The required electron-phonon spectrum has not been accepted.")
    inputs = root / stage["folder"] / "inputs"
    for name, digest in stage["metadata"]["input_sha256"].items():
        if _digest(inputs / name) != digest:
            raise ValueError(f"Prepared {name} changed. Prepare a new run.")
    dependencies = []
    inherited_pseudos = None
    parent_index = stage.get("restart_from")
    if parent_index is not None:
        parent = state["stages"][parent_index]
        result = parent.get("result") or {}
        if parent["status"] != "succeeded" or not result.get("success") or not result.get("checkpoint_sha256"):
            raise ValueError("The required QE checkpoint has not been accepted.")
        checkpoint = root / parent["folder"] / "outputs" / "checkpoint.json"
        if not checkpoint.is_file() or _digest(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("The accepted QE checkpoint receipt changed.")
        _validate_checkpoint(json.loads(checkpoint.read_text()), stage["name"])
        dependencies.append({"directory": state["remote_root"] + "/" + parent["folder"], "sha256": result["checkpoint_sha256"]})
        inherited_pseudos = json.loads((root / parent["folder"] / "inputs" / "pseudo_manifest.json").read_text())
    auxiliary_index = stage.get("auxiliary_from")
    if auxiliary_index is not None:
        from .qe_wannier import phonon_checkpoint_mapping
        if type(auxiliary_index) is not int or not 0 <= auxiliary_index < state["current_stage"]:
            raise ValueError("Invalid auxiliary phonon dependency.")
        auxiliary = state["stages"][auxiliary_index]
        accepted = auxiliary.get("result") or {}
        receipt = root / auxiliary["folder"] / "outputs" / "checkpoint.json"
        if (auxiliary["status"] != "succeeded" or not accepted.get("success")
                or not receipt.is_file() or _digest(receipt) != accepted.get("checkpoint_sha256")):
            raise ValueError("The auxiliary phonon checkpoint has not been accepted.")
        records = json.loads(receipt.read_text())
        _validate_checkpoint(records, "q2r")
        mapping = phonon_checkpoint_mapping(records, accepted.get("qpoint_count"))
        dependencies.append({"directory": state["remote_root"] + "/" + auxiliary["folder"],
                             "sha256": accepted["checkpoint_sha256"], "mapping": mapping})
    external = stage.get("external_checkpoint")
    if external:
        proof = root / external["receipt"]
        if not proof.resolve().is_relative_to(root) or _digest(proof) != external["receipt_sha256"]:
            raise ValueError("The approved continuation checkpoint receipt changed.")
        receipt = json.loads(proof.read_text())
        identity = {key: getattr(config, key) for key in ("host", "user", "port", "qe_command", "qe_bin_dir", "setup_commands")}
        if receipt["cluster"] != identity:
            raise ValueError("A saved QE checkpoint must use its original cluster and executable.")
        _validate_checkpoint(receipt["checkpoint"], stage["name"])
        dependencies.append({"directory": receipt["remote_directory"], "sha256": receipt["checkpoint_sha256"]})
        inherited_pseudos = receipt["pseudos"]
    remote = state["remote_root"] + "/" + stage["folder"]
    if stage["status"] == "planned":
        stage["status"] = state["status"] = "submitting"
        _save(root, state, f"Submitting QE {stage['name']}")
    if not stage["staged"]:
        _command(transport, "command -v sbatch >/dev/null && command -v squeue >/dev/null && command -v sacct >/dev/null && command -v python3 >/dev/null")
        pseudos = _pseudo_snapshot(transport, config, stage, inherited_pseudos)
        if inherited_pseudos is not None:
            # Older receipts predate the derived UPF type; the file identity
            # and every previously recorded header field must still match.
            comparable = {name: {key: value for key, value in entry.items()
                                if key != "pseudo_type" or key in inherited_pseudos.get(name, {})}
                          for name, entry in pseudos.items()}
            if comparable != inherited_pseudos:
                raise ValueError("Pseudopotentials changed since the accepted checkpoint. Prepare a fresh calculation.")
        _write(inputs / "pseudo_manifest.json", pseudos)
        runtime = {"input_sha256": stage["metadata"]["input_sha256"], "dependencies": dependencies,
                   "pseudo_root": config.qe_pseudo_root, "pseudos": pseudos}
        _write(inputs / "runtime.json", runtime)
        stage["runtime_sha256"] = _digest(inputs / "runtime.json")
        stage["pseudo_manifest_sha256"] = _digest(inputs / "pseudo_manifest.json")
        (inputs / "qe_runtime.py").write_text(_RUNTIME, encoding="utf-8")
        (inputs / "submit.sh").write_text(_script(config, stage), encoding="utf-8", newline="\n")
        names = [*stage["metadata"]["input_sha256"], "pseudo_manifest.json", "runtime.json", "qe_runtime.py", "submit.sh"]
        transport.upload_many([*(inputs / name for name in names), Path(__file__).with_name("dispatch.py")],
                              remote, permitted_names=_QE_UPLOAD_NAMES)
        stage["staged"] = True
        _save(root, state, "Uploaded QE inputs.")
    reply = _command(transport, f"python3 {shlex.quote(remote + '/dispatch.py')} {shlex.quote(remote)} {shlex.quote(stage['job_name'])}", timeout=100)
    receipt = json.loads(reply.splitlines()[-1])
    if receipt.get("job_name") != stage["job_name"] or not str(receipt.get("job_id", "")).isdigit():
        raise ValueError("Invalid Slurm submission receipt.")
    stage.update(job_id=str(receipt["job_id"]), status="queued", submitted_at=utc_now())
    state.update(status="queued", last_error=None, remote_failures=0)
    return _save(root, state, f"Slurm accepted job {stage['job_id']}")


def _remote_files(transport, remote):
    code = "from pathlib import Path;import json;root=Path(" + repr(remote) + ");paths=list(root.iterdir());\nfor relative in ('elph_dir','scratch/dft_agent.ephmat'):\n folder=root/relative\n if folder.is_dir() and not any(parent.is_symlink() for parent in (folder,*folder.parents)):\n  paths+=list(folder.iterdir())\nprint(json.dumps([p.relative_to(root).as_posix() for p in paths if p.is_file() and not p.is_symlink()]))"
    names = json.loads(_command(transport, "python3 -c " + shlex.quote(code)))
    return [_artifact_name(name) for name in names]


def _collect(root, state, transport, stage):
    remote = state["remote_root"] + "/" + stage["folder"]
    output = root / stage["folder"] / "outputs"
    allowed = {*RECEIPTS, *stage["metadata"]["input_sha256"],
               stage["metadata"]["output_file"], *stage["metadata"].get("output_files", [])}
    names = [name for name in _remote_files(transport, remote) if name in allowed or _RESULT_NAME.fullmatch(name) or _EPW_RESULT_NAME.fullmatch(name)]
    with tempfile.TemporaryDirectory(prefix=".qe-collect-", dir=output.parent) as directory:
        collected = Path(directory) / "outputs"
        collected.mkdir()
        manifest = transport.download_many(remote, collected, names, permitted_names=frozenset(names))
        if not isinstance(manifest, dict) or set(manifest) != set(names):
            raise ValueError("The collection receipt does not match the downloaded files.")
        for name, receipt in manifest.items():
            path = collected / name
            if (not _artifact_name(name) or path.is_symlink()
                    or not path.resolve().is_relative_to(collected.resolve()) or not path.is_file()
                    or not isinstance(receipt, dict) or type(receipt.get("size")) is not int
                    or path.stat().st_size != receipt["size"] or _digest(path) != receipt.get("sha256")):
                raise ValueError("A downloaded file does not match its collection receipt.")
        _write(collected / "artifact_manifest.json", manifest)
        shutil.copyfile(root / stage["folder"] / "inputs" / "metadata.json", collected / "metadata.json")
        result = _analyze_collection(root, stage, collected, manifest)
        _write(collected / "result.json", result)
        previous = Path(directory) / "previous"
        if output.exists():
            os.replace(output, previous)
        try:
            os.replace(collected, output)
        except OSError:
            if previous.exists():
                os.replace(previous, output)
            raise
    stage.update(result=result, status="succeeded" if result["success"] else "needs_attention")
    if result["success"] and not state["cancel_requested"] and state["current_stage"] + 1 < len(state["stages"]):
        state["current_stage"] += 1
        state["status"] = "planned"
    else:
        state["status"] = "cancelled" if state["cancel_requested"] else stage["status"]
    return _save(root, state, f"Collected QE {stage['name']}: {stage['status']}")


def _analyze_collection(root, stage, output, manifest):
    result = _science(stage["name"]).analyze_outputs(output, stage["name"], root / stage["folder"] / "inputs" / "structure.cif")
    reasons = [] if result.get("success") else [result.get("reason") or "QE output was not accepted."]
    expected = stage["metadata"]["input_sha256"]
    required = {*expected, "input_hashes.sha256", "execution.json", "runtime.json",
                "pseudo_manifest.json", stage["metadata"]["output_file"]}
    if not required <= set(manifest):
        reasons.append("Required collection files are missing: " + ", ".join(sorted(required - set(manifest))) + ".")
    recorded = {}
    if (output / "input_hashes.sha256").is_file():
        for line in (output / "input_hashes.sha256").read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                recorded[fields[1]] = fields[0]
    identity = recorded == expected and set(expected) <= set(manifest) and all((output / name).is_file() and _digest(output / name) == digest for name, digest in expected.items())
    result["input_identity_verified"] = identity
    if not identity:
        reasons.append("Input checksums do not match the approved inputs.")
    execution = json.loads((output / "execution.json").read_text()) if (output / "execution.json").is_file() else {}
    receipt_ok = (execution.get("job_id") == stage["job_id"]
                  and type(execution.get("exit_code")) is int
                  and execution.get("runtime_sha256") == stage.get("runtime_sha256")
                  and "runtime.json" in manifest
                  and _digest(output / "runtime.json") == stage.get("runtime_sha256")
                  and bool(execution.get("finished_at")))
    executed = receipt_ok and execution["exit_code"] == 0
    result["execution_verified"] = executed
    if not executed:
        reasons.append(f"QE exited with code {execution['exit_code']}." if receipt_ok
                       else "The job execution receipt is missing or inconsistent.")
    pseudo_ok = (output / "pseudo_manifest.json").is_file() and _digest(output / "pseudo_manifest.json") == stage.get("pseudo_manifest_sha256")
    result["pseudopotentials_verified"] = pseudo_ok
    if not pseudo_ok:
        reasons.append("The pseudopotential receipt does not match this run.")
    result["scheduler_state"] = stage.get("scheduler_state")
    if stage.get("scheduler_state") != "COMPLETED":
        reasons.append(f"Slurm ended in {stage.get('scheduler_state')}.")
    result["success"] = not reasons
    if reasons:
        result["reason"] = " ".join(reasons)
    if result["success"]:
        if (output / "checkpoint.json").is_file():
            result["checkpoint_sha256"] = _digest(output / "checkpoint.json")
        if (output / "final.cif").is_file():
            result["final_structure_sha256"] = _digest(output / "final.cif")
        if (output / "CONTCAR").is_file():
            result["final_native_structure_sha256"] = _digest(output / "CONTCAR")
    return result


def advance(run_dir, transport=None):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in TERMINAL:
            return state
        owned = transport is None
        stage = state["stages"][state["current_stage"]]
        try:
            _check_config(root, state)
            config = ClusterConfig.load(root / "config.json")
            transport = transport or SSHTransport(config)
            if stage["status"] in {"planned", "submitting"} and not state["cancel_requested"]:
                return _submit(root, state, config, transport)
            scheduler = _poll(transport, stage)
            stage["scheduler_state"] = scheduler
            if state["cancel_requested"] and scheduler not in SLURM_TERMINAL:
                _command(transport, "scancel " + stage["job_id"])
            if scheduler == "CANCELLED":
                stage["status"] = state["status"] = "cancelled"
                return _save(root, state, "Slurm confirmed cancellation.")
            if scheduler in SLURM_TERMINAL:
                stage["status"] = state["status"] = "collecting"
                _save(root, state)
                result = _collect(root, state, transport, stage)
                result["remote_failures"] = 0
                result["last_error"] = None
                return _save(root, result)
            if scheduler == "UNKNOWN":
                state["last_error"] = "Waiting for Slurm accounting; no resubmission performed."
            else:
                stage["status"] = state["status"] = "queued" if scheduler == "PENDING" else "running"
                state["last_error"] = None
            state["remote_failures"] = 0
            return _save(root, state)
        except TransportError as exc:
            state["last_error"] = str(exc)
            state["remote_failures"] = state.get("remote_failures", 0) + 1
            if state["remote_failures"] >= MAX_REMOTE_FAILURES or "outcome uncertain" in str(exc) or "did not acknowledge" in str(exc):
                state["status"] = "needs_attention"
            return _save(root, state, "Remote operation failed. Run details saved.")
        except (ValueError, OSError, KeyError, TypeError) as exc:
            state.update(status="needs_attention", last_error=str(exc))
            return _save(root, state, "Paused. Review the error before continuing.")
        finally:
            if owned and transport is not None:
                transport.close()


def resume(run_dir, transport=None):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] not in {"needs_attention", "failed"}:
            return state
        _check_config(root, state)
        stage = state["stages"][state["current_stage"]]
        owned = transport is None
        try:
            if not stage.get("job_id") and stage.get("staged"):
                config = ClusterConfig.load(root / "config.json")
                transport = transport or SSHTransport(config)
                remote = state["remote_root"] + "/" + stage["folder"]
                reply = _command(transport, f"python3 {shlex.quote(remote + '/dispatch.py')} {shlex.quote(remote)} {shlex.quote(stage['job_name'])} --reconcile-only", timeout=100)
                receipt = json.loads(reply.splitlines()[-1])
                if receipt.get("job_name") != stage["job_name"] or not str(receipt.get("job_id", "")).isdigit():
                    raise ValueError("Invalid reconciled job identity.")
                stage["job_id"] = str(receipt["job_id"])
            status = "queued" if stage.get("job_id") else "cancelled" if state.get("cancel_requested") else "planned"
            if status == "planned" and not stage.get("staged"):
                stage["materialized"] = False
            stage["status"] = state["status"] = status
            state.update(last_error=None, remote_failures=0)
            return _save(root, state, "Resumed the original QE run.")
        except (TransportError, ValueError, OSError, KeyError) as exc:
            state["last_error"] = str(exc)
            return _save(root, state, "Reconnection failed. No new job submitted.")
        finally:
            if owned and transport is not None:
                transport.close()


def cancel(run_dir, transport=None):
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        if state["status"] in {"succeeded", "cancelled"}:
            return state
        stage = state["stages"][state["current_stage"]]
        state["cancel_requested"] = True
        if not stage.get("job_id") and not stage.get("staged"):
            stage["status"] = state["status"] = "cancelled"
        elif not stage.get("job_id"):
            state.update(status="needs_attention", last_error="Reconcile the original submission before cancellation.")
        else:
            stage["status"] = state["status"] = "queued"
        _save(root, state, "Cancellation requested.")
    return advance(root, transport) if stage.get("job_id") else state


def bundle_run(run_dir):
    """Bundle inputs, receipts and collected raw data, excluding credentials."""
    root = Path(run_dir).expanduser().resolve()
    target = root / "results.zip"
    with zipfile.ZipFile(target.with_suffix(".tmp"), "w", zipfile.ZIP_DEFLATED) as archive:
        for name in ("run.json", "plan.json", "proposal.json", "explanations.json", "parent.json", "repair.json"):
            if (root / name).is_file():
                archive.write(root / name, name)
        state = read_state(root)
        for path in (root / "source").rglob("*") if (root / "source").is_dir() else []:
            if path.is_file() and not path.is_symlink() and path.name != "checkpoint.json":
                archive.write(path, path.relative_to(root))
        for stage in state["stages"]:
            for folder in ("inputs", "outputs"):
                source = root / stage["folder"] / folder
                for path in source.rglob("*") if source.is_dir() else []:
                    if path.is_file() and not path.is_symlink() and path.name not in {"runtime.json", "submit.sh", "qe_runtime.py"}:
                        archive.write(path, path.relative_to(root))
    os.replace(target.with_suffix(".tmp"), target)
    return target


def checkpoint_files(run_dir, stage_index=None):
    """List remote files recorded by an accepted QE stage."""
    root = Path(run_dir).expanduser().resolve()
    state = read_state(root)
    _check_config(root, state)
    index = state["current_stage"] if stage_index is None else stage_index
    if type(index) is not int or not 0 <= index < len(state["stages"]):
        raise ValueError("Choose an existing QE stage.")
    stage = state["stages"][index]
    if stage["status"] != "succeeded":
        raise ValueError("Download checkpoints only from an accepted stage.")
    path = root / stage["folder"] / "outputs" / "checkpoint.json"
    if _digest(path) != stage["result"]["checkpoint_sha256"]:
        raise ValueError("The checkpoint receipt changed.")
    return json.loads(path.read_text())


def fetch_raw_files(run_dir, names, stage_index=None, transport=None):
    """Fetch selected output/checkpoint files only on explicit request."""
    root = Path(run_dir).expanduser().resolve()
    with _lock(root):
        state = read_state(root)
        _check_config(root, state)
        index = state["current_stage"] if stage_index is None else stage_index
        stage = state["stages"][index]
        output = root / stage["folder"] / "outputs"
        checkpoint = checkpoint_files(root, index)
        if not names or any(name not in checkpoint for name in names):
            raise ValueError("Choose files listed in the accepted checkpoint receipt.")
        raw = output / "raw"
        receipt_path = raw / "artifact_manifest.json"

        def raw_path(name):
            relative = PurePosixPath(name)
            destination = raw / name
            if (not name or name == "artifact_manifest.json" or relative.is_absolute()
                    or ".." in relative.parts or raw.is_symlink() or destination.is_symlink()
                    or not destination.resolve().is_relative_to(raw.resolve())
                    or not raw.resolve().is_relative_to(output.resolve())):
                raise ValueError("Unsafe local checkpoint path.")
            return destination

        for name in names:
            raw_path(name)
        if receipt_path.is_symlink():
            raise ValueError("The raw download receipt is unsafe.")
        previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        if not isinstance(previous, dict):
            raise ValueError("The raw download receipt is invalid.")

        def verify_previous():
            if receipt_path.is_symlink():
                raise ValueError("The raw download receipt changed.")
            current = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
            if current != previous:
                raise ValueError("The raw download receipt changed.")
            for name, record in previous.items():
                path = raw_path(name)
                if (not isinstance(record, dict) or name not in checkpoint
                        or record != checkpoint[name] or not path.is_file()
                        or path.stat().st_size != record.get("size") or _digest(path) != record.get("sha256")):
                    raise ValueError("A previously downloaded checkpoint changed. Restore its verified file before downloading more.")

        verify_previous()
        config = ClusterConfig.load(root / "config.json")
        owned = transport is None
        transport = transport or SSHTransport(config)
        try:
            for name in names:
                relative = PurePosixPath(name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("Unsafe checkpoint path.")
            with tempfile.TemporaryDirectory(prefix=".qe-download-", dir=output) as directory:
                staging = Path(directory)
                manifest = transport.download_many(state["remote_root"] + "/" + stage["folder"], staging,
                                                   names, permitted_names=frozenset(names))
                if manifest != {name: checkpoint[name] for name in names}:
                    raise ValueError("Remote checkpoint no longer matches the accepted result.")
                verify_previous()
                for name in names:
                    staged = staging / name
                    if (staged.is_symlink() or not staged.resolve().is_relative_to(staging.resolve())
                            or not staged.is_file() or staged.stat().st_size != manifest[name]["size"]
                            or _digest(staged) != manifest[name]["sha256"]):
                        raise ValueError("The downloaded checkpoint does not match its receipt.")
                for name in names:
                    destination = raw_path(name)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(staging / name, destination)
            _write(receipt_path, {**previous, **manifest})
            return manifest
        finally:
            if owned:
                transport.close()
