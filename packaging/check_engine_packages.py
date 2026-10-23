"""Check that each pinned ABINIT, CP2K and Elk package set runs where engine_packages.json says it does.

Package metadata is not enough. Some conda-forge builds use newer glibc symbols than they
declare (CP2K 2026.2's libxs and libxsmm need glibc 2.38), and some link a library version
their dependency pins do not select (CP2K 2026.1 build 11 needs libdbcsr.so.2.9). For every
variant this downloads the packages, checks their SHA-256, reads the ELF files and reports:

- the newest GLIBC_ symbol version any file needs, which must not exceed the variant's min_glibc;
- shared libraries that the program or its MPI launcher loads, directly or indirectly, and
  that neither the package set nor glibc provides.

Usage: python packaging/check_engine_packages.py CACHE_DIR [ENGINE ...]
Reading .conda packages needs the zstandard module. The first run downloads about 3 GB.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import posixpath
import struct
import sys
import tarfile
import urllib.request
import zipfile

CATALOG = Path(__file__).resolve().parents[1] / "src" / "vasp_slurm_agent" / "engine_packages.json"
GLIBC = ("libc.so", "libm.so", "libpthread.so", "libdl.so", "librt.so", "ld-linux", "libutil.so", "libresolv.so",
         "libanl.so", "libmvec.so", "libnsl.so")
LAUNCHERS = {"mpich": ["bin/mpiexec.hydra", "bin/hydra_pmi_proxy"]}


def _sections(data):
    end = "<" if data[5] == 1 else ">"
    if data[4] == 2:
        offset, = struct.unpack_from(end + "Q", data, 0x28)
        size, count = struct.unpack_from(end + "HH", data, 0x3A)
        layout = end + "IIQQQQIIQQ"
    else:
        offset, = struct.unpack_from(end + "I", data, 0x20)
        size, count = struct.unpack_from(end + "HH", data, 0x2E)
        layout = end + "IIIIIIIIII"
    # (type, file offset, size, link, info) for each section header
    return end, [tuple(struct.unpack_from(layout, data, offset + i * size)[j] for j in (1, 4, 5, 6, 7)) for i in range(count)]


def _string(data, table, offset):
    start = table[1] + offset
    return data[start:data.index(b"\0", start)].decode()


def read_elf(data):
    """GLIBC_ versions the image requires, its DT_NEEDED libraries and its DT_SONAME."""
    end, sections = _sections(data)
    versions, needed, soname = set(), [], None
    for kind, offset, size, link, info in sections:
        if kind == 0x6FFFFFFE:  # SHT_GNU_verneed
            position = offset
            for _ in range(info):
                _, count, library, aux, following = struct.unpack_from(end + "HHIII", data, position)
                library = _string(data, sections[link], library)
                entry = position + aux
                for _ in range(count):
                    _, _, _, name, step = struct.unpack_from(end + "IHHII", data, entry)
                    name = _string(data, sections[link], name)
                    if name.startswith("GLIBC_2.") and library.startswith(GLIBC):
                        versions.add(name)
                    if not step:
                        break
                    entry += step
                if not following:
                    break
                position += following
        elif kind == 6:  # SHT_DYNAMIC
            entry, width = (end + "qQ", 16) if data[4] == 2 else (end + "iI", 8)
            for position in range(offset, offset + size, width):
                tag, value = struct.unpack_from(entry, data, position)
                if tag == 0:
                    break
                if tag == 1:
                    needed.append(_string(data, sections[link], value))
                elif tag == 14:
                    soname = _string(data, sections[link], value)
    return versions, needed, soname


def _members(path):
    def walk(tar):
        for member in tar:
            if member.isfile():
                yield member.name, tar.extractfile(member), None
            elif member.issym() or member.islnk():
                yield member.name, None, member.linkname
    if path.suffix == ".conda":
        import zstandard
        with zipfile.ZipFile(path) as archive:
            inner = next(name for name in archive.namelist() if name.startswith("pkg-") and name.endswith(".tar.zst"))
            with archive.open(inner) as compressed, tarfile.open(
                    fileobj=zstandard.ZstdDecompressor().stream_reader(compressed), mode="r|") as tar:
                yield from walk(tar)
    else:
        with tarfile.open(path, mode="r|bz2") as tar:
            yield from walk(tar)


def scan(path):
    report = {"glibc": {}, "elf": {}, "links": {}}
    for name, stream, link in _members(path):
        if stream is None:
            report["links"][name] = link
            continue
        head = stream.read(4)
        if head != b"\x7fELF":
            continue
        versions, needed, soname = read_elf(head + stream.read())
        if versions:
            report["glibc"][name] = max(versions, key=_version)
        report["elf"][name] = {"needed": needed, "soname": soname}
    return report


def _version(name):
    return tuple(int(part) for part in name.split("_", 1)[1].split("."))


def _digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(cache, package):
    target = cache / package["fn"]
    if not target.is_file() or _digest(target) != package["sha256"]:
        temporary = target.with_name(target.name + ".part")
        with urllib.request.urlopen(package["url"], timeout=120) as response, temporary.open("wb") as out:
            while block := response.read(1 << 20):
                out.write(block)
        if _digest(temporary) != package["sha256"]:
            temporary.unlink()
            raise ValueError("SHA-256 mismatch for " + package["fn"])
        temporary.replace(target)
    report = target.with_name(target.name + ".elf.json")
    if not report.is_file():
        report.write_text(json.dumps(scan(target)))
    return json.loads(report.read_text())


def unresolved(reports, programs):
    """Libraries the programs load that neither the package set nor glibc provides."""
    files = {name: info for report in reports for name, info in report["elf"].items()}
    links = {name: target for report in reports for name, target in report["links"].items()}
    names = {}
    for name, info in files.items():
        names.setdefault(posixpath.basename(name), name)
        if info["soname"]:
            names.setdefault(info["soname"], name)
    for name, target in links.items():
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(name), target))
        for _ in range(8):
            if resolved in files or resolved not in links:
                break
            resolved = posixpath.normpath(posixpath.join(posixpath.dirname(resolved), links[resolved]))
        if resolved in files:
            names.setdefault(posixpath.basename(name), resolved)
    missing = {program: ["(not in the package set)"] for program in programs if program not in files}
    queue, seen = [program for program in programs if program in files], set()
    while queue:
        current = queue.pop()
        if current in seen:
            continue
        seen.add(current)
        for library in files[current]["needed"]:
            if library.startswith(GLIBC):
                continue
            if library in names:
                queue.append(names[library])
            else:
                missing.setdefault(library, []).append(current)
    return missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("cache", type=Path)
    parser.add_argument("engines", nargs="*")
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    catalog = json.loads(CATALOG.read_text())
    failed = False
    for engine, entry in sorted(catalog["engines"].items()):
        if args.engines and engine not in args.engines:
            continue
        for platform, variants in sorted(entry["platforms"].items()):
            for key, variant in sorted(variants.items()):
                with ThreadPoolExecutor(max_workers=6) as pool:
                    reports = list(pool.map(lambda package: fetch(args.cache, package), variant["packages"]))
                needs = max((v for report in reports for v in report["glibc"].values()), key=_version, default="GLIBC_2.0")
                programs = ["bin/" + entry["program"], *LAUNCHERS.get(variant.get("mpi"), [])]
                missing = unresolved(reports, programs)
                ok = _version(needs) <= _version("GLIBC_" + variant["min_glibc"]) and not missing
                failed |= not ok
                print(json.dumps({"engine": engine, "version": variant.get("version", entry["version"]), "platform": platform,
                                  "min_glibc": variant["min_glibc"], "needs": needs.split("_", 1)[1],
                                  "missing_libraries": {k: v[:3] for k, v in missing.items()}, "ok": ok}), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
