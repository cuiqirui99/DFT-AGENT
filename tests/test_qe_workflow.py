from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

from vasp_slurm_agent.config import ClusterConfig
from vasp_slurm_agent.transport import Result
from vasp_slurm_agent import qe_workflow as workflow


class FakeTransport:
    def __init__(self):
        self.commands, self.uploads, self.downloads = [], [], []
        self.bulk_uploads, self.bulk_downloads = [], []
        self.jobs, self.files = {}, {}
        self.queue, self.accounting, self.exit_code = "PENDING", "", "0:0"
        self.lose_reply = False
        self.cancel_count = 0

    def run(self, command, timeout=60):
        self.commands.append(command)
        words = shlex.split(command)
        if words[:2] == ["python3", "-c"]:
            if "root.iterdir()" in words[2]:
                return Result(0, json.dumps(list(self.files)))
            return Result(0, json.dumps({"Si.UPF": {"size": 3, "sha256": hashlib.sha256(b"upf").hexdigest(), "element": "Si", "fully_relativistic": False, "functional": "PBE"}}))
        if command.startswith("python3 ") and "/dispatch.py" in command:
            name = words[-2] if "--reconcile-only" in words else words[-1]
            if "--reconcile-only" in words and name not in self.jobs:
                return Result(1, "", "No recorded submission")
            self.jobs.setdefault(name, str(7000 + len(self.jobs)))
            if self.lose_reply:
                self.lose_reply = False
                return Result(255, "", "Connection lost after submission")
            return Result(0, json.dumps({"job_id": self.jobs[name], "job_name": name}))
        if command.startswith("squeue "):
            return Result(0, self.queue)
        if command.startswith("sacct "):
            job = words[words.index("-j") + 1]
            return Result(0, f"{job}|{self.accounting}|{self.exit_code}|" if self.accounting else "")
        if command.startswith("scancel "):
            self.cancel_count += 1
            return Result(0)
        if command.startswith("command -v ") or command.startswith("mkdir -p "):
            return Result(0)
        pytest.fail(f"Unexpected command: {command}")

    def upload(self, source, destination):
        self.uploads.append(destination)
        data = Path(source).read_bytes()
        return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}

    def upload_many(self, sources, remote, *, permitted_names):
        self.bulk_uploads.append((list(sources), remote, permitted_names))
        assert isinstance(permitted_names, frozenset)
        assert {Path(source).name for source in sources} <= permitted_names
        return {Path(source).name: self.upload(source, remote + "/" + Path(source).name) for source in sources}

    def download_many(self, remote, folder, names, *, permitted_names):
        self.bulk_downloads.append((remote, folder, names, permitted_names))
        assert isinstance(permitted_names, frozenset) and set(names) <= permitted_names
        result = {}
        for name in names:
            target = Path(folder) / name
            target.parent.mkdir(parents=True, exist_ok=True)
            result[name] = self.download(remote + "/" + name, target)
        return result

    def download(self, source, destination):
        self.downloads.append(source)
        name = next(name for name in self.files if source.endswith("/" + name))
        data = self.files[name]
        Path(destination).write_bytes(data)
        return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


@pytest.fixture
def config():
    return ClusterConfig(engine="qe", host="cluster.invalid", user="test",
                         remote_root="/scratch/test", partition="cpu", qe_pseudo_root="/pseudo",
                         qe_pseudopotentials={"Si": "Si.UPF"}, qe_command="mpirun -np 8 pw.x")


@pytest.fixture
def prepared(tmp_path, monkeypatch, config):
    source = tmp_path / "Si.cif"
    source.write_text("original structure")

    def build(source, folder, task, parameters, potentials):
        folder.mkdir(parents=True, exist_ok=True)
        program = workflow.PROGRAMS[task]
        stem = "pw" if program == "pw.x" else program[:-2]
        (folder / (stem + ".in")).write_text(f"task={task}\nstructure={Path(source).read_text()}")
        (folder / "structure.cif").write_text(Path(source).read_text())
        (folder / "qe-input.json").write_text(json.dumps({"task": task}))
        return {"formula": "Si", "input_file": stem + ".in", "output_file": stem + ".out",
                "input_files": [stem + ".in", "structure.cif"], "output_files": [stem + ".out"],
                "pseudo_files": ["Si.UPF"] if program == "pw.x" else []}

    def analyze(folder, task, expected):
        (folder / "final.cif").write_text("accepted structure")
        return {"success": True, "reason": "mock parser accepted", "artifacts": []}

    monkeypatch.setattr(workflow.qe, "prepare_inputs", build)
    monkeypatch.setattr(workflow.qe, "analyze_outputs", analyze)
    monkeypatch.setattr(workflow, "SSHTransport", lambda *args: pytest.fail("Unexpected real connection"))

    def prepare(tasks=("scf",), parameters=None, stage_parameters=None):
        root = tmp_path / ("run-" + str(len(list(tmp_path.glob("run-*")))))
        workflow.prepare_plan(source, root, config, list(tasks), parameters, stage_parameters=stage_parameters)
        return root
    return prepare


def complete(root, transport):
    state = workflow.read_state(root)
    stage = state["stages"][state["current_stage"]]
    inputs = root / stage["folder"] / "inputs"
    transport.files = {name: (inputs / name).read_bytes() for name in stage["metadata"]["input_sha256"]}
    transport.files.update({
        "input_hashes.sha256": "".join(digest + "  " + name + "\n" for name, digest in stage["metadata"]["input_sha256"].items()).encode(),
        "execution.json": json.dumps({"job_id": stage["job_id"], "exit_code": 0,
            "runtime_sha256": stage["runtime_sha256"], "finished_at": "2026-10-14T12:00:00Z"}).encode(),
        "pseudo_manifest.json": (inputs / "pseudo_manifest.json").read_bytes(),
        "runtime.json": (inputs / "runtime.json").read_bytes(),
        "checkpoint.json": b"{}", stage["metadata"]["output_file"]: b"JOB DONE.\n",
    })
    transport.queue, transport.accounting = "", "COMPLETED"


def test_dependencies_expand_before_any_remote_work(prepared):
    root = prepared(["relax", "bands", "dos", "pdos", "phonon_bands"])
    state = workflow.read_state(root)
    assert state["engine"] == "qe"
    assert state["status"] == "planned"
    assert [s["name"] for s in state["stages"]] == ["relax", "scf", "bands", "nscf", "dos", "projwfc", "ph", "q2r", "matdyn"]
    assert all(s["structure_from"] == 0 for s in state["stages"][1:])
    assert state["stages"][6]["parameters"]["qmesh"] == [2, 2, 2]
    assert state["stages"][7]["restart_from"] == 6
    assert not any(s["job_id"] for s in state["stages"])


def test_changed_method_gets_its_own_charge_density(prepared):
    root = prepared(["scf", "bands"], stage_parameters=[{"functional": "PBE"}, {"functional": "HSE06"}])
    stages = workflow.read_state(root)["stages"]
    assert [s["name"] for s in stages] == ["scf", "scf", "bands"]
    assert stages[2]["restart_from"] == 1


def test_input_and_config_tampering_block_before_remote_commands(prepared):
    root = prepared()
    (root / "01_scf" / "inputs" / "pw.in").write_text("changed")
    transport = FakeTransport()
    state = workflow.advance(root, transport)
    assert state["status"] == "needs_attention"
    assert not transport.commands
    root = prepared()
    path = root / "config.json"
    value = json.loads(path.read_text())
    value["host"] = "changed.invalid"
    path.write_text(json.dumps(value))
    assert workflow.advance(root, transport)["status"] == "needs_attention"
    assert not transport.commands


def test_lost_reply_reconciles_original_submission(prepared):
    root = prepared()
    transport = FakeTransport()
    transport.lose_reply = True
    first = workflow.advance(root, transport)
    assert first["status"] == "submitting"
    assert first["stages"][0]["staged"]
    assert len(transport.jobs) == 1
    uploads = len(transport.uploads)
    second = workflow.advance(root, transport)
    assert second["status"] == "queued"
    assert len(transport.jobs) == 1
    assert len(transport.uploads) == uploads


def test_accounting_lag_does_not_resubmit(prepared):
    root, transport = prepared(), FakeTransport()
    workflow.advance(root, transport)
    transport.queue = ""
    state = workflow.advance(root, transport)
    assert state["status"] == "queued"
    assert "accounting" in state["last_error"]
    assert len(transport.jobs) == 1


@pytest.mark.parametrize("failure", ["scheduler", "input", "execution", "pseudo", "parser"])
def test_independent_acceptance_gates(prepared, monkeypatch, failure):
    root, transport = prepared(["scf", "bands"]), FakeTransport()
    workflow.advance(root, transport)
    complete(root, transport)
    if failure == "scheduler":
        transport.accounting = "TIMEOUT"
    elif failure == "input":
        transport.files["pw.in"] = b"changed"
    elif failure == "execution":
        value = json.loads(transport.files["execution.json"])
        value["job_id"] = "999999"
        transport.files["execution.json"] = json.dumps(value).encode()
    elif failure == "pseudo":
        transport.files["pseudo_manifest.json"] = b"{}"
    else:
        monkeypatch.setattr(workflow.qe, "analyze_outputs", lambda *args: {"success": False, "reason": "not converged"})
    state = workflow.advance(root, transport)
    assert state["status"] == "needs_attention"
    assert state["current_stage"] == 0
    assert len(transport.jobs) == 1
    assert not state["stages"][0]["result"]["success"]


def test_accepted_relaxation_feeds_later_inputs(prepared):
    root, transport = prepared(["relax", "bands"]), FakeTransport()
    workflow.advance(root, transport)
    complete(root, transport)
    state = workflow.advance(root, transport)
    assert state["status"] == "planned" and state["current_stage"] == 1
    transport.queue, transport.accounting = "PENDING", ""
    state = workflow.advance(root, transport)
    assert state["status"] == "queued"
    assert "accepted structure" in (root / "02_scf" / "inputs" / "pw.in").read_text()
    assert not any("wfc" in path for path in transport.downloads)


def test_solver_failure_is_distinct_from_missing_execution_receipt(prepared):
    root, transport = prepared(), FakeTransport()
    workflow.advance(root, transport)
    complete(root, transport)
    receipt = json.loads(transport.files["execution.json"])
    receipt["exit_code"] = 1
    transport.files["execution.json"] = json.dumps(receipt).encode()
    transport.accounting = "FAILED"
    state = workflow.advance(root, transport)
    result = state["stages"][0]["result"]
    assert state["status"] == "needs_attention"
    assert not result["success"] and not result["execution_verified"]
    assert "QE exited with code 1" in result["reason"]
    assert "receipt is missing" not in result["reason"]


def test_cancel_finishing_stage_never_starts_dependency(prepared):
    root, transport = prepared(["scf", "bands"]), FakeTransport()
    workflow.advance(root, transport)
    complete(root, transport)
    state = workflow.cancel(root, transport)
    assert state["status"] == "cancelled"
    assert state["current_stage"] == 0
    assert len(transport.jobs) == 1


def test_cancel_before_submission_is_local(prepared):
    root, transport = prepared(), FakeTransport()
    assert workflow.cancel(root, transport)["status"] == "cancelled"
    assert not transport.commands


def test_remote_traversal_is_rejected(prepared):
    root, transport = prepared(), FakeTransport()
    workflow.advance(root, transport)
    complete(root, transport)
    transport.files["../other"] = b"anything"
    state = workflow.advance(root, transport)
    assert state["status"] == "needs_attention"
    assert not transport.downloads


def test_launch_preserves_mpi_and_replaces_only_binary(config):
    assert workflow._command_for(config, "ph.x") == "mpirun -np 8 ph.x"
    from dataclasses import replace
    other = replace(config, qe_bin_dir="/opt/qe/bin")
    assert workflow._command_for(other, "matdyn.x") == "mpirun -np 8 /opt/qe/bin/matdyn.x"
    with pytest.raises(ValueError):
        workflow._command_for(config, "bash")


def runtime_fixture(tmp_path):
    root = tmp_path / "job"
    root.mkdir()
    pseudo = tmp_path / "pseudo"
    pseudo.mkdir()
    (pseudo / "Si.UPF").write_bytes(b"upf")
    (root / "pw.in").write_bytes(b"input")
    spec = {"input_sha256": {"pw.in": hashlib.sha256(b"input").hexdigest()}, "dependencies": [],
            "pseudo_root": str(pseudo), "pseudos": {"Si.UPF": {"sha256": hashlib.sha256(b"upf").hexdigest(), "size": 3}}}
    (root / "runtime.json").write_text(json.dumps(spec))
    (root / "qe_runtime.py").write_text(workflow._RUNTIME)
    return root, pseudo, spec


def test_runtime_checks_inputs_pseudos_and_forbids_second_execution(tmp_path):
    root, pseudo, spec = runtime_fixture(tmp_path)
    run = lambda: subprocess.run([sys.executable, "qe_runtime.py", "prepare"], cwd=root, capture_output=True, text=True)
    assert run().returncode == 0
    assert (root / "pseudo" / "Si.UPF").read_bytes() == b"upf"
    assert run().returncode != 0


def test_runtime_rejects_changed_pseudo(tmp_path):
    root, pseudo, spec = runtime_fixture(tmp_path)
    (pseudo / "Si.UPF").write_bytes(b"changed")
    result = subprocess.run([sys.executable, "qe_runtime.py", "prepare"], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Pseudopotential changed" in result.stderr


def test_runtime_preserves_nested_phonon_checkpoint_and_verifies_hash(tmp_path):
    root, pseudo, spec = runtime_fixture(tmp_path)
    source = tmp_path / "previous"
    file = source / "scratch" / "_ph0" / "dft_agent.phsave" / "control_ph.xml"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"phonon state")
    relative = file.relative_to(source).as_posix()
    receipt = source / "checkpoint.json"
    receipt.write_text(json.dumps({relative: {"size": file.stat().st_size, "sha256": workflow._digest(file)}}))
    spec["dependencies"] = [{"directory": str(source), "sha256": workflow._digest(receipt)}]
    (root / "runtime.json").write_text(json.dumps(spec))
    result = subprocess.run([sys.executable, "qe_runtime.py", "prepare"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / relative).read_bytes() == b"phonon state"
    (root / ".execution_started").unlink()
    file.write_bytes(b"changed")
    result = subprocess.run([sys.executable, "qe_runtime.py", "prepare"], cwd=root, capture_output=True, text=True)
    assert result.returncode != 0
    assert "Accepted checkpoint changed" in result.stderr


def test_runtime_finish_copies_xml_and_records_checkpoint(tmp_path, monkeypatch):
    root, pseudo, spec = runtime_fixture(tmp_path)
    monkeypatch.setenv("SLURM_JOB_ID", "7744")
    subprocess.run([sys.executable, "qe_runtime.py", "prepare"], cwd=root, check=True)
    xml = root / "scratch" / "dft_agent.save" / "data-file-schema.xml"
    xml.parent.mkdir(parents=True)
    xml.write_bytes(b"<xml/>")
    result = subprocess.run([sys.executable, "qe_runtime.py", "finish", "0"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (root / "data-file-schema.xml").read_bytes() == b"<xml/>"
    execution = json.loads((root / "execution.json").read_text())
    assert execution["exit_code"] == 0 and execution["job_id"] == "7744"
    receipt = json.loads((root / "checkpoint.json").read_text())
    assert receipt["scratch/dft_agent.save/data-file-schema.xml"]["sha256"] == workflow._digest(xml)


def test_bundle_contains_raw_results_without_config(prepared):
    import zipfile
    root, transport = prepared(), FakeTransport()
    workflow.advance(root, transport)
    complete(root, transport)
    assert workflow.advance(root, transport)["status"] == "succeeded"
    from vasp_slurm_agent.workflow import bundle_run
    archive = bundle_run(root)
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        assert "config.json" not in names
        assert "source/structure.cif" in names
        assert "01_scf/outputs/pw.out" in names
        assert "01_scf/outputs/input_hashes.sha256" in names
        assert not any(name.endswith("runtime.json") for name in names)


class LocalHeaderTransport:
    def __init__(self, local_root):
        self.local_root = local_root

    def run(self, command, timeout=60):
        words = shlex.split(command)
        assert words[:2] == ["python3", "-c"]
        assert words[2].count("root=Path('/pseudo')") == 1
        code = words[2].replace("root=Path('/pseudo')", f"root=Path({str(self.local_root)!r})")
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        return Result(result.returncode, result.stdout, result.stderr)


def test_submission_checks_real_upf_element_soc_and_known_hash(tmp_path, config):
    from dataclasses import replace
    from vasp_slurm_agent.transport import TransportError
    pseudo = tmp_path / "Si.UPF"
    def write(element="Si", soc=False):
        pseudo.write_text(f'<UPF><PP_HEADER element="{element}" functional="PBE" has_so="{"T" if soc else "F"}"/><PP_MESH>' + " " * 220 + '</PP_MESH><PP_LOCAL>0</PP_LOCAL></UPF>')
    write()
    cfg = config
    stage = {"metadata": {"pseudo_files": ["Si.UPF"]}, "parameters": {}}
    receipt = workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)
    assert receipt["Si.UPF"]["element"] == "Si"
    assert receipt["Si.UPF"]["sha256"] == workflow._digest(pseudo)
    write(element="Al")
    with pytest.raises(TransportError, match="does not match"):
        workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)
    write()
    cfg = replace(cfg, qe_pseudopotentials={"Si": {"filename": "Si.UPF", "fully_relativistic": True}})
    stage["parameters"]["soc"] = True
    with pytest.raises(TransportError, match="spin-orbit data"):
        workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)
    write(soc=True)
    assert workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)["Si.UPF"]["fully_relativistic"]
    cfg = replace(cfg, qe_pseudopotentials={"Si": {"filename": "Si.UPF", "sha256": "0" * 64}})
    with pytest.raises(ValueError, match="validated library"):
        workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)


def test_submission_checks_actual_pp_type_and_functional(tmp_path, config):
    from dataclasses import replace
    pseudo = tmp_path / "Si.UPF"
    def write(pseudo_type):
        pseudo.write_text(f'<UPF><PP_HEADER element="Si" functional="PBE" pseudo_type="{pseudo_type}" has_so="F"/><PP_MESH>' + " " * 220 + '</PP_MESH><PP_LOCAL>0</PP_LOCAL></UPF>')
    # User-entered metadata cannot override the actual UPF header.
    cfg = replace(config, qe_pseudopotentials={
        "Si": {"filename": "Si.UPF", "pseudo_type": "NC", "functional": "LDA"}})
    stage = {"name": "relax", "metadata": {"pseudo_files": ["Si.UPF"]}, "parameters": {"functional": "HSE06"}}
    write("PAW")
    with pytest.raises(ValueError, match="norm-conserving"):
        workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)
    stage["name"] = "scf"
    assert workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)["Si.UPF"]["pseudo_type"] == "PAW"
    stage["name"] = "relax"
    write("NC")
    assert workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)["Si.UPF"]["pseudo_type"] == "NC"
    stage["parameters"]["spin"] = "noncollinear"
    stage["metadata"]["namelists"] = {"control": {"tstress": True}}
    with pytest.raises(ValueError, match="stress or cell relaxation"):
        workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)
    stage["metadata"]["namelists"]["control"]["tstress"] = False
    assert workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)["Si.UPF"]["pseudo_type"] == "NC"
    stage["metadata"].pop("namelists")
    write("")
    with pytest.raises(ValueError, match="verified norm-conserving"):
        workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)
    stage["parameters"] = {"functional": "LDA"}
    with pytest.raises(ValueError, match="compatible LDA UPFs"):
        workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)

    stage.update(name="wannier_scf", parameters={})
    write("PAW")
    with pytest.raises(ValueError, match="norm-conserving"):
        workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)
    write("NC")
    assert workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)["Si.UPF"]["pseudo_type"] == "NC"


def test_submission_rejects_no_soc_from_actual_full_uspp_header(tmp_path, config):
    from dataclasses import replace
    pseudo = tmp_path / "Si.UPF"
    pseudo.write_text('<UPF><PP_HEADER element="Si" functional="PBE" pseudo_type="US" has_so="T"/><PP_MESH>'
                      + " " * 220 + '</PP_MESH><PP_LOCAL>0</PP_LOCAL></UPF>')
    # A stale scalar/NC configuration must not bypass the actual file check.
    cfg = replace(config, qe_pseudopotentials={
        "Si": {"filename": "Si.UPF", "fully_relativistic": False, "pseudo_type": "NC"}})
    stage = {"name": "scf", "metadata": {"pseudo_files": ["Si.UPF"]}, "parameters": {"soc": False}}
    with pytest.raises(ValueError, match="requires SOC"):
        workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)
    stage["parameters"] = {"namelists": {"system": {"noncolin": True, "lspinorb": True}}}
    receipt = workflow._pseudo_snapshot(LocalHeaderTransport(tmp_path), cfg, stage)
    assert receipt["Si.UPF"]["fully_relativistic"] is True and receipt["Si.UPF"]["pseudo_type"] == "US"


def test_downstream_submission_requires_real_checkpoint_members(prepared):
    root, transport = prepared(["scf", "bands"]), FakeTransport()
    workflow.advance(root, transport)
    complete(root, transport)
    assert workflow.advance(root, transport)["status"] == "planned"
    previous_jobs = len(transport.jobs)
    state = workflow.advance(root, transport)
    assert state["status"] == "needs_attention"
    assert "checkpoint is empty" in state["last_error"]
    assert len(transport.jobs) == previous_jobs


def test_checkpoint_requirements_match_solver_stage():
    entry = {"sha256": "0" * 64, "size": 10}
    records = {"scratch/dft_agent.save/data-file-schema.xml": entry,
               "scratch/dft_agent.save/charge-density.dat": entry}
    workflow._validate_checkpoint(records, "bands")
    with pytest.raises(ValueError, match="wavefunction"):
        workflow._validate_checkpoint(records, "ph")
    records["scratch/dft_agent.save/wfc1.dat"] = entry
    workflow._validate_checkpoint(records, "ph")
    with pytest.raises(ValueError, match="dense electronic"):
        workflow._validate_checkpoint(records, "eph_ph")
    records["scratch/dft_agent.a2Fsave"] = entry
    workflow._validate_checkpoint(records, "eph_ph")
    with pytest.raises(ValueError, match="dynamical"):
        workflow._validate_checkpoint(records, "q2r")


@pytest.mark.parametrize("changed_field", [None, "sha256", "size", "element", "functional", "fully_relativistic"])
def test_checkpoint_survives_added_upf_type_but_not_changed_identity(prepared, monkeypatch, changed_field):
    root, transport = prepared(["scf", "bands"]), FakeTransport()
    workflow.advance(root, transport)
    complete(root, transport)
    record = {"sha256": "0" * 64, "size": 10}
    transport.files["checkpoint.json"] = json.dumps({
        "scratch/dft_agent.save/data-file-schema.xml": record,
        "scratch/dft_agent.save/charge-density.dat": record}).encode()
    assert workflow.advance(root, transport)["status"] == "planned"
    receipt_path = root / "01_scf" / "inputs" / "pseudo_manifest.json"
    original = receipt_path.read_bytes()
    current = json.loads(original)
    current["Si.UPF"]["pseudo_type"] = "US"
    if changed_field:
        current["Si.UPF"][changed_field] = "changed"
    monkeypatch.setattr(workflow, "_pseudo_snapshot", lambda *args: current)
    state = workflow.advance(root, transport)
    assert receipt_path.read_bytes() == original
    if changed_field:
        assert state["status"] == "needs_attention"
        assert "Pseudopotentials changed" in state["last_error"]
        assert len(transport.jobs) == 1
    else:
        assert state["status"] == "queued"
        assert len(transport.jobs) == 2
        assert json.loads((root / "02_bands" / "inputs" / "pseudo_manifest.json").read_text()) == current


def test_electron_phonon_chain_is_ordered_and_cannot_start_midway():
    stages = workflow._sequence(["electron_phonon"], {}, None)
    assert [stage["name"] for stage in stages] == ["eph_dense_scf", "eph_scf", "eph_ph", "eph_q2r", "eph_matdyn", "eph_lambda"]
    assert [stage["restart_from"] for stage in stages] == [None, 0, 1, 2, 3, 4]
    assert all(stage["request_index"] == 0 for stage in stages)
    with pytest.raises(ValueError, match="complete validated dependency"):
        workflow._sequence(["eph_ph"], {}, None)


def test_lambda_uses_serial_stdin_and_does_not_reinterpret_mpi_arguments(config):
    stage = {"name": "eph_lambda", "metadata": {"input_file": "lambda.in", "output_file": "lambda.out", "no_mpi": True}}
    script = workflow._script(config, stage)
    assert "lambda.x < lambda.in > lambda.out" in script
    assert "mpirun" not in script


def test_workflow_batches_inputs_and_only_present_results(prepared):
    root, transport = prepared(), FakeTransport()
    state = workflow.advance(root, transport)
    assert state["status"] == "queued"
    assert len(transport.bulk_uploads) == 1
    assert len(transport.bulk_uploads[0][0]) == 8
    complete(root, transport)
    transport.files["unrelated.txt"] = b"not selected"
    state = workflow.advance(root, transport)
    assert state["status"] == "succeeded"
    assert len(transport.bulk_downloads) == 1
    downloaded = transport.bulk_downloads[0][2]
    assert "pw.out" in downloaded and "unrelated.txt" not in downloaded
    assert "slurm.err" not in downloaded  # Optional files absent on the host stay optional.


def test_qe_gpu_is_rejected_before_preparation_or_submission(tmp_path, config, prepared):
    from dataclasses import replace
    gpu = replace(config, compute_backend="gpu", tasks=1, gpus_per_node=1)
    with pytest.raises(ValueError, match="GPU execution has not been verified"):
        workflow.prepare_plan(tmp_path / "missing.cif", tmp_path / "new", gpu, ["scf"])
    root = prepared()
    with pytest.raises(ValueError, match="GPU execution has not been verified"):
        workflow.attach_config(root, gpu)
    state = workflow.read_state(root)
    gpu.save(root / "config.json")
    state["config_sha256"] = workflow._digest(root / "config.json")
    workflow._write(root / "run.json", state)
    transport = FakeTransport()
    assert workflow.advance(root, transport)["status"] == "needs_attention"
    assert not transport.commands and not transport.jobs


@pytest.mark.parametrize("fault", [None, "changed", "overwrite", "escape"])
def test_wannier_auxiliary_copy_keeps_nscf_and_checks_derivatives(tmp_path, fault):
    root, _, spec = runtime_fixture(tmp_path)
    source = tmp_path / "phonon"
    derivative = source / "scratch/_ph0/dft_agent.dvscf1"
    derivative.parent.mkdir(parents=True)
    derivative.write_bytes(b"accepted derivative")
    native = derivative.relative_to(source).as_posix()
    records = {native: {"size": derivative.stat().st_size, "sha256": workflow._digest(derivative)}}
    receipt = source / "checkpoint.json"
    receipt.write_text(json.dumps(records))
    target = "save/dft_agent.dvscf_q1" if fault != "escape" else "save/../../escape"
    spec["dependencies"] = [{"directory": str(source), "sha256": workflow._digest(receipt),
                             "mapping": {native: target}}]
    wavefunction = root / "scratch/dft_agent.save/wfc1.dat"
    wavefunction.parent.mkdir(parents=True)
    wavefunction.write_bytes(b"nscf wavefunction")
    if fault == "changed":
        derivative.write_bytes(b"changed derivative")
    if fault == "overwrite":
        (root / target).parent.mkdir(parents=True)
        (root / target).write_bytes(b"existing")
    (root / "runtime.json").write_text(json.dumps(spec))
    result = subprocess.run([sys.executable, "qe_runtime.py", "prepare"], cwd=root,
                            capture_output=True, text=True)
    assert (result.returncode == 0) == (fault is None), result.stderr
    assert wavefunction.read_bytes() == b"nscf wavefunction"
    if fault is None:
        assert (root / target).read_bytes() == b"accepted derivative"
    elif fault == "overwrite":
        assert (root / target).read_bytes() == b"existing"
