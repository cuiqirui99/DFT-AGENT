from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from vasp_slurm_agent import qe_setup as setup
from vasp_slurm_agent.transport import Result

pytestmark = pytest.mark.skipif(os.name == "nt", reason="Executes remote POSIX build and Slurm scripts locally; covered on Linux/macOS.")


def config(tmp_path, **changes):
    values = dict(remote_root=str(tmp_path / "remote"), setup_commands=[], partition="test", account="", qe_bin_dir="", qe_pseudo_root=str(tmp_path / "pseudos"))
    return SimpleNamespace(**(values | changes))


class LocalTransport:
    def __init__(self):
        self.uploads = []
        self.bulk_uploads = []
        self.commands = []
        self.https_payloads = {}

    def run(self, command, timeout=60):
        self.commands.append(command)
        if "urllib.request" in command:
            code = shlex.split(command)[-1]
            # Keep remote HTTPS tests offline while exercising the actual helper.
            prefix = f'''import io,urllib.request
payloads={self.https_payloads!r}
class Response(io.BytesIO):
    def geturl(self): return self.url
def fetch(request,**kwargs):
    if request.full_url not in payloads: raise OSError('Network unavailable')
    response=Response(payloads[request.full_url]); response.url=request.full_url
    return response
urllib.request.urlopen=fetch
'''
            command = "python3 -c " + shlex.quote(prefix + code)
        result = subprocess.run(["bash", "-c", command], capture_output=True, text=True, timeout=timeout)
        return Result(result.returncode, result.stdout, result.stderr)

    def upload(self, source, destination):
        self.uploads.append(destination)
        Path(destination).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)

    def upload_many(self, sources, destination, *, permitted_names):
        assert isinstance(permitted_names, frozenset)
        assert {Path(path).name for path in sources} <= permitted_names
        self.bulk_uploads.append(([Path(path).name for path in sources], destination))
        result = {}
        for source in sources:
            self.upload(source, str(Path(destination) / Path(source).name))
            result[Path(source).name] = {"sha256": setup._digest(Path(source)), "size": Path(source).stat().st_size}
        return result


def upf(element="Si"):
    return (f'<UPF version="2.0.1"><PP_HEADER element="{element}" functional="PBE" relativistic="scalar" has_so="F"/><PP_MESH>1 2</PP_MESH><PP_LOCAL>1 2</PP_LOCAL>' + " " * 200 + "</UPF>").encode()


def test_plan_reproducible_and_cpu_build_not_calculation_resources(tmp_path):
    cfg = config(tmp_path)
    plan = setup.install_plan(cfg, build_jobs=6)
    assert plan == setup.install_plan(cfg, prefix=plan["prefix"], build_jobs=6)
    script = setup._build_script(plan)
    assert "#SBATCH --ntasks=1" in script
    assert "#SBATCH --cpus-per-task=6" in script
    assert "python3 build.py" in script
    assert "srun" not in script
    assert "lambda.x" in plan["programs"]
    assert plan["sources"]["qe-7.4.1.tar.gz"]["sha256"] == "6ef9c53dbf0add2a5bf5ad2a372c0bff935ad56c4472baa001003e4f932cab97"
    assert plan["build_targets"] == ["pw", "ph", "pp", "neb"]
    assert "wannier90.tar.gz" not in plan["sources"]


def test_epw_plan_pins_dependency_and_has_separate_installation(tmp_path):
    cfg = config(tmp_path)
    core = setup.install_plan(cfg)
    plan = setup.install_plan(cfg, include_epw=True)
    assert plan == setup.install_plan(cfg, prefix=plan["prefix"], include_epw=True)
    assert plan["prefix"] != core["prefix"]
    assert plan["programs"] == [*setup.REQUIRED_PROGRAMS, "epw.x", "wannier90.x"]
    assert plan["build_targets"] == [*setup.BUILD_TARGETS, "epw"]
    assert plan["build_steps"] == [list(setup.BUILD_TARGETS), ["epw"]]
    assert plan["sources"]["wannier90.tar.gz"]["sha256"] == "bc93ba6774ecc41aa766303747ef355849d63f469c2a437d03acda4e98515725"
    assert plan["sources"]["wannier90.tar.gz"]["source_archive_sha256"] == "e5936d20c4e19f458335a53a1ec94ac993e54b09ee0e44b3184ac13c0f2758ad"
    assert "1d6b187374a2d50b509e5e79e2cab01a79ff7ce1" in plan["sources"]["wannier90.tar.gz"]["url"]
    assert "wannier90.tar.gz" not in core["sources"]
    with pytest.raises(ValueError, match="include EPW"):
        setup.install_plan(cfg, include_epw="yes")


@pytest.mark.parametrize("prefix", ["/", "/usr/local", "relative", "/safe/../etc", "/bad\npath"])
def test_install_rejects_unsafe_prefix(tmp_path, prefix):
    with pytest.raises(ValueError):
        setup.install_plan(config(tmp_path), prefix=prefix)


def test_tampered_plan_is_not_submitted(tmp_path):
    cfg = config(tmp_path)
    plan = setup.install_plan(cfg)
    plan["sources"] = {"bad": {"url": "https://example.com/", "sha256": "0" * 64}}
    transport = LocalTransport()
    with pytest.raises(ValueError, match="changed"):
        setup.start_install(cfg, plan, transport)
    assert not transport.uploads


def test_changed_build_targets_are_not_submitted(tmp_path):
    cfg = config(tmp_path)
    plan = setup.install_plan(cfg)
    plan["build_targets"] = ["all"]
    transport = LocalTransport()
    with pytest.raises(ValueError, match="plan changed"):
        setup.start_install(cfg, plan, transport)
    assert not transport.commands and not transport.uploads


def test_nested_source_edit_cannot_change_the_pinned_catalogue(tmp_path):
    cfg = config(tmp_path)
    plan = setup.install_plan(cfg, include_epw=True)
    original = setup.EPW_SOURCES["wannier90.tar.gz"]["sha256"]
    plan["sources"]["wannier90.tar.gz"]["sha256"] = "0" * 64
    plan["sources"]["wannier90.tar.gz"]["include_paths"].append("examples")
    assert setup.EPW_SOURCES["wannier90.tar.gz"]["sha256"] == original
    assert "examples" not in setup.EPW_SOURCES["wannier90.tar.gz"]["include_paths"]
    transport = LocalTransport()
    with pytest.raises(ValueError, match="plan changed"):
        setup.start_install(cfg, plan, transport)
    assert not transport.commands


def test_pseudo_plan_cutoffs_and_soc_boundary():
    plan = setup.plan_pseudopotentials(["Sr", "Ti", "O", "Ti"])
    assert len(plan["elements"]) == 3
    assert plan["ecutwfc_ry"] == 50
    assert plan["ecutrho_ry"] == 400
    assert plan["elements"]["Sr"]["license"] == "GPL-3.0"
    assert plan["elements"]["O"]["license"] == "GPL-2.0-or-later"
    # Spin-orbit coupling selects the fully relativistic PseudoDojo set; it has no Ni.
    soc = setup.plan_pseudopotentials(["Bi", "Se"], soc=True)
    assert soc["library_id"] == "pseudodojo-fr" and soc["relativistic"] == "full"
    assert soc["elements"]["Bi"]["filename"] == "Bi.pbe-nc-fr-04-pd.upf" and soc["elements"]["Bi"]["url"].endswith("/Bi/Bi-d_r.upf")
    with pytest.raises(ValueError, match="No PseudoDojo fully relativistic pseudopotential for Ni"):
        setup.plan_pseudopotentials(["Ni"], soc=True)
    with pytest.raises(ValueError, match="No SSSP"):
        setup.plan_pseudopotentials(["Bogus"])


def test_nc_library_is_explicit_complete_and_converts_hartree_hints():
    catalogue = json.loads(Path(setup.__file__).with_name("qe_nc_pseudos.json").read_text())
    plan = setup.plan_pseudopotentials(["Al", "Mg", "B"], library="pseudodojo")
    assert len(catalogue["elements"]) == 72
    assert plan["pseudo_type"] == "NC"
    assert plan["ecutwfc_ry"] == 84
    assert plan["ecutrho_ry"] == 336
    assert plan["elements"]["Al"]["cutoff_wfc"] == 40
    assert plan["elements"]["B"]["cutoff_wfc"] == 76
    assert plan["elements"]["Mg"]["valence_electrons"] == 10
    assert plan["license"]["name"] == "CC-BY-4.0"
    assert plan["cutoff_source"]["hints_unit"] == "Hartree"
    assert plan["archive_sha256"] == "603d7f07d364b60b321aa5c8eb0ae809aab339c3e884e4b62f31c35370991067"
    for element, item in catalogue["elements"].items():
        assert item["filename"] == element + ".pbe-nc-sr-04-pd.upf"
        assert item["pseudo_type"] == "NC" and len(item["sha256"]) == 64
        assert 0 < item["cutoff_wfc"] * 4 == item["cutoff_rho"]
    assert setup.plan_pseudopotentials(["Al"])["library"].startswith("SSSP")
    assert setup.plan_pseudopotentials(["Al"], library="pseudodojo", soc=True)["library_id"] == "pseudodojo-fr"
    fr = json.loads(Path(setup.__file__).with_name("qe_nc_fr_pseudos.json").read_text())
    assert len(fr["elements"]) == 67 and set(fr["missing"]) == {"Lu", "Ni", "Pd", "Pt"}
    for element, item in fr["elements"].items():
        assert item["filename"] == element + ".pbe-nc-fr-04-pd.upf" and item["relativistic"] == "full"
        assert item["url"].startswith("https://raw.githubusercontent.com/PseudoDojo/ONCVPSP-PBE-FR-PDv0.4/") and len(item["sha256"]) == 64
        assert 0 < item["cutoff_wfc"] * 4 == item["cutoff_rho"]
    with pytest.raises(ValueError, match="No PseudoDojo"):
        setup.plan_pseudopotentials(["U"], library="pseudodojo")
    with pytest.raises(ValueError, match="Choose SSSP"):
        setup.plan_pseudopotentials(["Al"], library="unrecognized")


@pytest.mark.parametrize("pseudo_type", ["NC", "PAW"])
@pytest.mark.parametrize("remote_https", [False, True])
def test_nc_install_checks_type_and_keeps_existing_library(tmp_path, monkeypatch, pseudo_type, remote_https):
    data = upf("Al").replace(b'functional="PBE"', f'functional="PBE" pseudo_type="{pseudo_type}"'.encode())
    archive = tmp_path / "nc.tgz"
    with tarfile.open(archive, "w:gz") as out:
        member = tarfile.TarInfo("Al.upf")
        member.size = len(data)
        out.addfile(member, io.BytesIO(data))
    license_path = tmp_path / "PseudoDojo-README.md"
    license_path.write_text("Test attribution")
    plan = setup.plan_pseudopotentials(["Al"], library="pseudodojo")
    plan["archive_sha256"] = setup._digest(archive)
    plan["elements"]["Al"]["sha256"] = hashlib.sha256(data).hexdigest()
    plan["license"]["sha256"] = setup._digest(license_path)
    monkeypatch.setattr(setup, "plan_pseudopotentials", lambda *a, **kw: plan)
    monkeypatch.setattr(setup, "_download", lambda name, *args: license_path if name.endswith(".md") else archive)
    cfg = config(tmp_path)
    old = Path(cfg.qe_pseudo_root, "Al.pbe-sssp.UPF")
    old.parent.mkdir()
    old.write_bytes(b"Existing user library")
    transport = LocalTransport()
    if remote_https:
        transport.https_payloads[plan["source_url"]] = archive.read_bytes()
    if pseudo_type == "PAW":
        with pytest.raises(ValueError, match="not a scalar-relativistic PBE NC"):
            setup.ensure_pseudos(cfg, ["Al"], transport, library="pseudodojo")
        assert not transport.uploads
    else:
        result = setup.ensure_pseudos(cfg, ["Al"], transport, library="pseudodojo")
        installed = Path(cfg.qe_pseudo_root, "Al.pbe-nc-sr-04-pd.upf")
        assert installed.read_bytes() == data
        if remote_https:
            assert not any("Al.pbe-nc-sr-04-pd.upf" in name for name in transport.uploads)
        assert result["config_pseudopotentials"]["Al"]["pseudo_type"] == "NC"
        assert result["config_pseudopotentials"]["Al"]["ecutwfc"] == 40
        manifests = list(old.parent.glob("pseudodojo-*.json"))
        assert len(manifests) == 1
        assert json.loads(manifests[0].read_text())["license"]["name"] == "CC-BY-4.0"
        uploads = len(transport.uploads)
        setup.ensure_pseudos(cfg, ["Al"], transport, library="pseudodojo")
        assert len(transport.uploads) == uploads
    assert old.read_bytes() == b"Existing user library"


@pytest.mark.parametrize("has_so", [True, False])
def test_fully_relativistic_install_downloads_pinned_files_and_checks_spin_orbit(tmp_path, monkeypatch, has_so):
    data = upf("Bi").replace(b'functional="PBE"', b'functional="PBE" pseudo_type="NC"')
    if has_so:
        data = data.replace(b'relativistic="scalar" has_so="F"', b'relativistic="full" has_so="T"')
    source = tmp_path / "Bi-d_r.upf"
    source.write_bytes(data)
    license_path = tmp_path / "PseudoDojo-README.md"
    license_path.write_text("Test attribution")
    plan = setup.plan_pseudopotentials(["Bi"], soc=True)
    plan["elements"]["Bi"]["sha256"] = hashlib.sha256(data).hexdigest()
    plan["license"]["sha256"] = setup._digest(license_path)
    monkeypatch.setattr(setup, "plan_pseudopotentials", lambda *a, **kw: plan)
    fetched = []
    monkeypatch.setattr(setup, "_download", lambda name, metadata, cache: fetched.append(metadata.get("url")) or (license_path if name.endswith(".md") else source))
    cfg = config(tmp_path)
    transport = LocalTransport()
    if not has_so:
        with pytest.raises(ValueError, match="not a fully relativistic PBE NC"):
            setup.ensure_pseudos(cfg, ["Bi"], transport, soc=True)
        return
    result = setup.ensure_pseudos(cfg, ["Bi"], transport, soc=True)
    assert Path(cfg.qe_pseudo_root, "Bi.pbe-nc-fr-04-pd.upf").read_bytes() == data
    assert result["config_pseudopotentials"]["Bi"]["fully_relativistic"] is True
    assert plan["elements"]["Bi"]["url"] in fetched


def test_upf_headers_reject_wrong_element_and_entities():
    assert setup.validate_upf(upf(), "Si")["has_so"] is False
    with pytest.raises(ValueError, match="does not match"):
        setup.validate_upf(upf(), "Fe")
    with pytest.raises(ValueError, match="entities"):
        setup.validate_upf(b'<!ENTITY x SYSTEM "file:///etc/passwd">' + upf(), "Si")
    with pytest.raises(ValueError, match="complete UPF"):
        setup.validate_upf(b"<html>" + b" " * 300, "Si")
    full = upf().replace(b'relativistic="scalar" has_so="F"', b'relativistic="full" has_so="T"')
    assert setup.validate_upf(full, "Si")["has_so"] is True


def test_upf_legacy_header():
    data = ("<PP_HEADER>\n0 Version Number\nSr Element\nUS Ultrasoft\nT Core\nSLA PW PBE PBE Exchange-Correlation\n10.0 Zval\n</PP_HEADER><PP_MESH></PP_MESH><PP_LOCAL></PP_LOCAL>" + " " * 100).encode()
    assert setup.validate_upf(data, "Sr")["element"] == "Sr"
    assert setup.validate_upf(data, "Sr")["valence_electrons"] == 10.0


def test_upf_valence_is_read_for_empty_band_counts():
    assert setup.validate_upf(upf(), "Si")["valence_electrons"] is None
    data = upf().replace(b'has_so="F"', b'has_so="F" z_valence=" 4.000000000000000D+00"')
    assert setup.validate_upf(data, "Si")["valence_electrons"] == 4.0
    # Every bundled library entry carries the valence of its file.
    for name in ("qe_pseudos.json", "qe_nc_pseudos.json", "qe_nc_fr_pseudos.json"):
        catalogue = json.loads(Path(setup.__file__).with_name(name).read_text())
        entries = catalogue.get("elements", catalogue)
        assert len(entries) > 60 and all(0 < entry["valence_electrons"] < 100 for entry in entries.values())


def test_download_checksum_failure_keeps_old_cache(tmp_path, monkeypatch):
    target = tmp_path / "file.tar.gz"
    target.write_bytes(b"old")

    class Download(io.BytesIO):
        def geturl(self):
            return "https://example.org/file"

    monkeypatch.setattr(setup.urllib.request, "urlopen", lambda *a, **kw: Download(b"wrong"))
    with pytest.raises(ValueError, match="SHA-256"):
        setup._download(target.name, {"url": "https://example.org/file", "sha256": "0" * 64}, tmp_path)
    assert target.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [target]


def test_unreachable_download_names_the_host_and_the_cache(tmp_path, monkeypatch):
    # www.pseudo-dojo.org timed out on 2026-10-22 from both the laptop and Arrhenius.
    def offline(*args, **kwargs):
        raise setup.urllib.error.URLError(TimeoutError(60, "Operation timed out"))

    monkeypatch.setattr(setup.urllib.request, "urlopen", offline)
    with pytest.raises(ValueError, match=r"www\.pseudo-dojo\.org did not deliver nc\.tgz .*Try again later, or put a copy with SHA-256 0{64}") as error:
        setup._download("nc.tgz", {"url": "https://www.pseudo-dojo.org/pseudos/nc.tgz", "sha256": "0" * 64}, tmp_path)
    assert str(tmp_path) in str(error.value) and list(tmp_path.iterdir()) == []


def test_build_archive_keeps_verified_source_and_license_without_datasets(tmp_path):
    original = tmp_path / "upstream-wannier90.tar.gz"
    entries = {"Makefile": b"build", "src/a.f90": b"program", "LICENSE": b"license", "examples/large.dat": b"large dataset"}
    with tarfile.open(original, "w:gz") as archive:
        for name, data in entries.items():
            member = tarfile.TarInfo("upstream/" + name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    metadata = {"url": "https://example.org/wannier.tar.gz", "include_paths": ["Makefile", "src", "LICENSE"],
                "source_archive_sha256": setup._digest(original)}
    derived = tmp_path / "derived.tar.gz"
    setup._build_source_archive(original, derived, metadata)
    metadata["sha256"] = setup._digest(derived)
    installed = setup._download("wannier90.tar.gz", metadata, tmp_path)
    assert installed.read_bytes() == derived.read_bytes()
    with tarfile.open(installed) as archive:
        names = set(archive.getnames())
        assert names == {"source/Makefile", "source/src/a.f90", "source/LICENSE", "source/.dft-agent-source.json"}
        manifest = json.load(archive.extractfile("source/.dft-agent-source.json"))
        assert manifest["files"] == {name: hashlib.sha256(data).hexdigest() for name, data in entries.items() if not name.startswith("examples/")}
        assert manifest["source_archive_sha256"] == setup._digest(original)
    original.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="original build source checksum"):
        setup._build_source_archive(original, derived, metadata)


@pytest.mark.parametrize("include_epw", [False, True])
def test_start_is_idempotent_and_records_uncertain_submission(tmp_path, monkeypatch, include_epw):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    calls = tmp_path / "sbatch-calls"
    for name in ("sbatch", "make", "git", "mpif90", "squeue", "sacct", "scancel"):
        path = fakebin / name
        path.write_text(f"#!/bin/sh\necho called >> '{calls}'\necho '1234;cluster'\n" if name == "sbatch" else "#!/bin/sh\nexit 0\n")
        path.chmod(0o700)
    monkeypatch.setenv("PATH", str(fakebin) + ":" + __import__("os").environ["PATH"])
    sources = {name: {"url": "https://example.org/" + name, "sha256": hashlib.sha256(name.encode()).hexdigest()}
               for name in ("qe-7.4.1.tar.gz", "devxlib.tar.gz", "fox.tar.gz", "lapack.tar.gz", "mbd.tar.gz")}
    monkeypatch.setattr(setup, "SOURCES", sources)
    epw_sources = {"wannier90.tar.gz": {"url": "https://example.org/wannier90.tar.gz",
                   "sha256": hashlib.sha256(b"wannier90.tar.gz").hexdigest()}}
    monkeypatch.setattr(setup, "EPW_SOURCES", epw_sources)
    expected_sources = sources | (epw_sources if include_epw else {})
    cache = tmp_path / "cache"
    cache.mkdir()
    for name in expected_sources:
        (cache / name).write_bytes(name.encode())
    cfg = config(tmp_path)
    plan = setup.install_plan(cfg, include_epw=include_epw)
    transport = LocalTransport()
    receipt = setup.start_install(cfg, plan, transport, cache_dir=cache)
    assert receipt["job_id"] == "1234"
    assert len(transport.bulk_uploads) == 1
    assert set(transport.bulk_uploads[0][0]) == {*expected_sources, "plan.json", "build.py", "build.sh", "dispatch.py", "scheduler.py"}
    assert len(transport.commands) == 4  # Receipt, prerequisites, all HTTPS sources, publish/submit.
    assert not list(Path(plan["work_dir"]).glob(".stage-*"))
    for name in expected_sources:
        assert Path(plan["work_dir"], name).read_bytes() == name.encode()
    assert setup.start_install(cfg, plan, transport, cache_dir=cache) == receipt
    assert calls.read_text().splitlines() == ["called"]
    Path(plan["work_dir"], "submission.json").write_text(json.dumps({"status": "submitting"}))
    assert setup.start_install(cfg, plan, transport, cache_dir=cache)["status"] == "submitting"
    assert calls.read_text().splitlines() == ["called"]


def test_source_staging_prefers_remote_https_and_transfers_only_missing(tmp_path, monkeypatch):
    work, cache = tmp_path / "work", tmp_path / "cache"
    work.mkdir()
    cache.mkdir()
    payloads = {"cached.tar.gz": b"existing", "remote.tar.gz": b"remote download", "local.tar.gz": b"local fallback"}
    sources = {name: {"url": "https://example.org/" + name, "sha256": hashlib.sha256(data).hexdigest()} for name, data in payloads.items()}
    (work / "cached.tar.gz").write_bytes(payloads["cached.tar.gz"])
    (cache / "local.tar.gz").write_bytes(payloads["local.tar.gz"])
    transport = LocalTransport()
    transport.https_payloads[sources["remote.tar.gz"]["url"]] = payloads["remote.tar.gz"]
    paths = setup._source_transfers(transport, sources, str(work), cache)
    assert paths == [cache / "local.tar.gz"]
    assert len(transport.commands) == 1 and not transport.uploads
    assert (work / "remote.tar.gz").read_bytes() == payloads["remote.tar.gz"]
    assert (work / "cached.tar.gz").read_bytes() == payloads["cached.tar.gz"]
    assert not (work / "local.tar.gz").exists()


def test_source_staging_rejects_bad_https_checksum_without_fallback(tmp_path):
    sources = {"source.tar.gz": {"url": "https://example.org/source", "sha256": hashlib.sha256(b"correct").hexdigest()}}
    transport = LocalTransport()
    transport.https_payloads["https://example.org/source"] = b"corrupt"
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "source.tar.gz").write_bytes(b"correct")
    with pytest.raises(setup.TransportError, match="checksum mismatch"):
        setup._source_transfers(transport, sources, str(tmp_path / "work"), cache)
    assert not transport.uploads
    assert not list((tmp_path / "work").iterdir())


def test_source_staging_bounds_a_stalled_download(tmp_path):
    class StalledTransport:
        def run(self, command, timeout=60):
            code = shlex.split(command)[-1].replace("process.join(20)", "process.join(0.1)")
            prefix = "import time,urllib.request\ndef stalled(*a,**k): time.sleep(30)\nurllib.request.urlopen=stalled\n"
            result = subprocess.run(["python3", "-c", prefix + code], capture_output=True, text=True, timeout=5)
            return Result(result.returncode, result.stdout, result.stderr)

    cache = tmp_path / "cache"
    cache.mkdir()
    path = cache / "source.tar.gz"
    path.write_bytes(b"cached source")
    sources = {path.name: {"url": "https://example.org/source", "sha256": setup._digest(path)}}
    assert setup._source_transfers(StalledTransport(), sources, str(tmp_path / "work"), cache) == [path]
    assert not list((tmp_path / "work").iterdir())


def test_ensure_pseudos_checks_before_writing_and_does_not_replace(tmp_path, monkeypatch):
    data = upf()
    filename = "Si.test.UPF"
    digest = hashlib.sha256(data).hexdigest()
    archive = tmp_path / "archive.tar.gz"
    with tarfile.open(archive, "w:gz") as out:
        info = tarfile.TarInfo(filename)
        info.size = len(data)
        out.addfile(info, io.BytesIO(data))
    license_path = tmp_path / "SSSP-LICENSE.txt"
    license_path.write_text("test license")
    monkeypatch.setattr(setup, "PSEUDO_LICENSE", {"url": "https://example.org/license", "sha256": hashlib.sha256(license_path.read_bytes()).hexdigest()})
    monkeypatch.setattr(setup, "_download", lambda name, *args: license_path if name.endswith(".txt") else archive)
    monkeypatch.setattr(setup, "plan_pseudopotentials", lambda *a, **kw: {"elements": {"Si": {"filename": filename, "sha256": digest}}})
    cfg = config(tmp_path)
    transport = LocalTransport()
    result = setup.ensure_pseudos(cfg, ["Si"], transport, cache_dir=tmp_path)
    installed = Path(cfg.qe_pseudo_root, filename)
    assert installed.read_bytes() == data
    assert result["pseudopotentials"] == {"Si": filename}
    uploads = len(transport.uploads)
    setup.ensure_pseudos(cfg, ["Si"], transport, cache_dir=tmp_path)
    assert len(transport.uploads) == uploads
    installed.write_bytes(b"user custom file")
    with pytest.raises(ValueError, match="different file"):
        setup.ensure_pseudos(cfg, ["Si"], transport, cache_dir=tmp_path)
    assert installed.read_bytes() == b"user custom file"


def test_poll_completed_without_manifest_is_failure(tmp_path, monkeypatch):
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    for name, output in (("squeue", ""), ("sacct", "COMPLETED")):
        path = fakebin / name
        path.write_text(f"#!/bin/sh\necho '{output}'\n")
        path.chmod(0o700)
    monkeypatch.setenv("PATH", str(fakebin) + ":" + __import__("os").environ["PATH"])
    result = setup.poll_install(config(tmp_path), {"work_dir": str(tmp_path), "job_id": "123", "status": "submitted"}, LocalTransport())
    assert result["status"] == "failed"
    assert "manifest" in result["error"]


def test_build_requires_slurm_before_changing_files(tmp_path):
    plan = setup.install_plan(config(tmp_path))
    script = tmp_path / "build.py"
    script.write_text(setup._BUILD_PY)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    env = dict(__import__("os").environ)
    env.pop("SLURM_JOB_ID", None)
    result = subprocess.run([__import__("sys").executable, str(script)], env=env, capture_output=True, text=True)
    assert result.returncode != 0
    assert "must run in Slurm" in result.stderr
    assert not Path(plan["prefix"]).exists()
    assert json.loads((tmp_path / "build-status.json").read_text())["status"] == "failed"


@pytest.mark.parametrize("include_epw", [False, True])
@pytest.mark.parametrize("failure", [None, "checksum", "targets", "missing_executable"])
def test_build_stages_only_required_dependencies_and_verifies_all_programs(tmp_path, failure, include_epw):
    dependencies = ("devxlib", "fox", "lapack", "mbd") + (("wannier90",) if include_epw else ())
    plan = setup.install_plan(config(tmp_path), include_epw=include_epw)
    configure = (
        "#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n"
        "prefix=next(arg.split('=',1)[1] for arg in sys.argv if arg.startswith('--prefix='))\n"
        "Path('make.inc').write_text('# __MPI\\nPREFIX = '+prefix+'\\n')\n"
    )
    # The native submodule helper skips network access only after a .git marker
    # exists. Exercise this requirement before the build and install commands.
    checks = "\n".join(
        f"\ttest -e external/{name}/.git || git fetch unavailable\n"
        f"\ttest -f external/{name}/source.f90" for name in dependencies
    )
    missing = "epw.x" if include_epw else "lambda.x"
    installed = [name for name in plan["programs"] if not (failure == "missing_executable" and name == missing)]
    makefile = ("include make.inc\nall:\n\tgit fetch unused-wannier-dependency\n"
        "pw ph pp neb epw:\n" + checks + "\n"
        "\tif [ \"$@\" = epw ]; then test \"$(MAKECMDGOALS)\" = epw -a -e compiled-ph; fi\n"
        "\ttouch compiled-$@\ninstall:\n"
        "\ttest " + " -a ".join("-e compiled-" + target for target in plan["build_targets"]) + "\n"
        "\tmkdir -p \"$(PREFIX)/bin\"\n" +
        "".join(f'\tcp executable "$(PREFIX)/bin/{name}"\n' for name in installed))
    for name in plan["sources"]:
        entries = ({"configure": configure, "Makefile": makefile,
                    "executable": "#!/bin/sh\nexit 0\n"}
                   if name == "qe-7.4.1.tar.gz" else {"source.f90": "! pinned dependency\n"})
        if "source_archive_sha256" in plan["sources"][name]:
            entries[".dft-agent-source.json"] = json.dumps({
                "source_archive_sha256": plan["sources"][name]["source_archive_sha256"],
                "files": {key: hashlib.sha256(value.encode()).hexdigest() for key, value in entries.items()}})
        path = tmp_path / name
        with tarfile.open(path, "w:gz") as archive:
            for filename, content in entries.items():
                item = tarfile.TarInfo("source/" + filename)
                data = content.encode()
                item.size, item.mode = len(data), 0o755
                archive.addfile(item, io.BytesIO(data))
        plan["sources"] = {**plan["sources"], name: {**plan["sources"][name], "sha256": setup._digest(path)}}
    if failure == "checksum":
        (tmp_path / "fox.tar.gz").write_bytes(b"corrupt source")
    if failure == "targets":
        plan["build_targets"] = ["all"]
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    for name in ("git", "curl", "wget"):
        command = fakebin / name
        command.write_text("#!/bin/sh\necho unexpected-network-command >&2\nexit 97\n")
        command.chmod(0o755)
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    script = tmp_path / "build.py"
    script.write_text(setup._BUILD_PY)
    env = dict(os.environ, SLURM_JOB_ID="test", MPIF90="mpif90", PATH=str(fakebin) + ":" + os.environ["PATH"])
    result = subprocess.run([__import__("sys").executable, str(script)], env=env, capture_output=True, text=True, timeout=20)
    status = json.loads((tmp_path / "build-status.json").read_text())
    if failure:
        assert result.returncode != 0 and status["status"] == "failed"
        error = {"checksum": "Source checksum mismatch: fox.tar.gz", "targets": "Unsupported QE build targets",
                 "missing_executable": "Missing executable: " + missing}[failure]
        assert error in result.stderr
        if failure != "missing_executable":
            assert not (tmp_path / "source").exists()
        assert not Path(plan["prefix"]).exists()
    else:
        assert result.returncode == 0, result.stderr
        assert status["status"] == "installed"
        manifest = json.loads(Path(plan["prefix"], "dft-agent-install.json").read_text())
        assert set(manifest["executable_sha256"]) == set(plan["programs"])
        assert manifest["build_targets"] == plan["build_targets"]
        assert set(manifest["sources"]) == {"qe-7.4.1.tar.gz", *(name + ".tar.gz" for name in dependencies)}
        assert (tmp_path / "source" / "external" / "wannier90").exists() is include_epw
    assert "unexpected-network-command" not in result.stderr


@pytest.mark.parametrize("changed_name", ["source.tar.gz", "build.py"])
def test_installer_never_overwrites_a_file_changed_during_staging(tmp_path, monkeypatch, changed_name):
    data = b"pinned source"
    sources = {"source.tar.gz": {"url": "https://example.org/source", "sha256": hashlib.sha256(data).hexdigest()}}
    monkeypatch.setattr(setup, "SOURCES", sources)
    monkeypatch.setattr(setup, "inspect", lambda *args: {"tools": {name: "available" for name in ("sbatch", "squeue", "sacct", "scancel", "make", "git", "mpif90", "mpifort", "mpiifort")}})
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "source.tar.gz").write_bytes(data)
    cfg = config(tmp_path)
    plan = setup.install_plan(cfg)
    class ChangedDuringUpload(LocalTransport):
        def upload_many(self, sources, destination, *, permitted_names):
            receipt = super().upload_many(sources, destination, permitted_names=permitted_names)
            (Path(destination).parent / changed_name).write_bytes(b"keep existing content")
            return receipt
    with pytest.raises(setup.TransportError, match="Existing installer file differs"):
        setup.start_install(cfg, plan, ChangedDuringUpload(), cache_dir=cache)
    assert Path(plan["work_dir"], changed_name).read_bytes() == b"keep existing content"
    assert not Path(plan["work_dir"], "submission.json").exists()
    assert not list(Path(plan["work_dir"]).glob(".stage-*"))


def test_pbs_installer_recovers_lost_ack_without_second_build(tmp_path, monkeypatch):
    """Offline qsub/qstat executables model server receipts; QE is not built."""
    from vasp_slurm_agent.scheduler import get_scheduler
    fakebin = tmp_path / "bin"
    fakebin.mkdir()
    calls = tmp_path / "submissions"
    qsub = fakebin / "qsub"
    qsub.write_text(f"#!/bin/sh\necho submitted >> '{calls}'\necho '1234.pbs.example'\n")
    qsub.chmod(0o700)
    monkeypatch.setenv("PATH", str(fakebin) + ":" + os.environ["PATH"])
    source = tmp_path / "fixture.tar.gz"
    source.write_bytes(b"source fixture")
    monkeypatch.setattr(setup, "SOURCES", {source.name: {"url": "https://example.invalid/source", "sha256": setup._digest(source)}})
    monkeypatch.setattr(setup, "_source_transfers", lambda *args: [source])
    tools = {name: "available" for name in ("qsub", "qstat", "qdel", "make", "git", "mpif90", "mpifort", "mpiifort")}
    monkeypatch.setattr(setup, "inspect", lambda *args: {"tools": tools})
    cfg = config(tmp_path, scheduler="pbs", extra_pbs=["-l mem=4gb"])
    plan = setup.install_plan(cfg)
    qstat = fakebin / "qstat"
    payload = json.dumps({"Jobs": {"1234.pbs.example": {"Job_Name": get_scheduler("pbs").job_name("qe-build-" + plan["id"]), "job_state": "F", "Exit_status": 0}}})
    qstat.write_text("#!/bin/sh\nprintf '%s\\n' " + shlex.quote(payload) + "\n")
    qstat.chmod(0o700)
    wire = LocalTransport()
    receipt = setup.start_install(cfg, plan, wire, cache_dir=tmp_path)
    assert receipt["job_id"] == "1234.pbs.example" and receipt["scheduler"] == "pbs"
    work = Path(plan["work_dir"])
    (work / "job.json").unlink()
    interrupted = {key: value for key, value in receipt.items() if key != "job_id"}
    interrupted["status"] = "submitting"
    (work / "submission.json").write_text(json.dumps(interrupted))
    recovered = setup.start_install(cfg, plan, wire, cache_dir=tmp_path)
    assert recovered["job_id"] == receipt["job_id"]
    assert calls.read_text().splitlines() == ["submitted"]
    assert setup.poll_install(cfg, recovered, wire)["status"] == "failed"  # No installed manifest.


def test_toolchain_keeps_tested_module_environment_and_rejects_shell_names(tmp_path):
    commands = []
    def run(command, timeout=60):
        commands.append(command)
        if '__TC_MODULES__' in command:
            return Result(0, '__TC_PATH__\nmake=/usr/bin/make\ngit=/usr/bin/git\n__TC_MODULES__\nopenmpi/4.1 (D)\nmpi;touch /tmp/bad\n__TC_END__\n')
        assert command.startswith('set -e\nmodule load site\nmodule load openmpi/4.1 ')
        assert 'module purge' not in command
        return Result(0, 'MPI=/modules/mpi/bin/mpifort\nMAKE=yes\nGIT=yes\n')
    cfg = config(tmp_path, setup_commands=['module load site'])
    found = setup.detect_toolchain(cfg, SimpleNamespace(run=run))
    assert found['ready']
    assert found['setup_commands'] == ['module load site', 'module load openmpi/4.1']
    assert len(commands) == 2
    assert all('touch' not in command for command in commands)


def test_toolchain_rejects_failed_existing_setup(tmp_path):
    cfg = config(tmp_path, setup_commands=['module load missing'])
    def run(command, timeout=60):
        assert command.startswith('set -e\nmodule load missing\n')
        return Result(1, '__TC_PATH__\nmpifort=/old/bin/mpifort\nmake=/usr/bin/make\n__TC_END__\n')
    with pytest.raises(setup.TransportError, match='environment setup'):
        setup.detect_toolchain(cfg, SimpleNamespace(run=run))
