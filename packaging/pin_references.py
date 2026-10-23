"""Pin the documentation files of an ABINIT or Quantum ESPRESSO release in references.json.

DFT Agent downloads these programs' input documentation, tutorials and example inputs on first
use and checks each file against the size and SHA-256 recorded here. This lists the files of a
release commit on GitHub, downloads them once and writes their pins.

Usage: python packaging/pin_references.py ENGINE VERSION COMMIT [--default]
ABINIT's GitHub mirror marks releases with commits titled "vX.Y.Z" rather than tags; QE's tags
(qe-7.6) resolve to their commits.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request

CATALOG = Path(__file__).resolve().parents[1] / "src" / "vasp_slurm_agent" / "references.json"
PATTERNS = {
    # Variables of abinit and anaddb, every tutorial and topic page, the guides of the programs DFT
    # Agent runs, and the inputs of the ground-state and response-function tutorials.
    "abinit": (r"abimkdocs/variables_(?:abinit|anaddb)\.py", r"doc/tutorial/[^/]+\.md", r"doc/topics/_[^/]+\.md",
               r"doc/guide/(?:abinit|anaddb|mrgddb|respfn|new_user)\.md", r"tests/(?:tutorial|tutorespfn)/Input/[^/]+\.abi"),
    # The input descriptions of the programs DFT Agent runs, the descriptions and scripts of the
    # official examples, and the EPW example inputs.
    "qe": (r"(?:PW/Doc/INPUT_PW|PP/Doc/INPUT_(?:BANDS|DOS|PROJWFC|PP)|PHonon/Doc/INPUT_(?:PH|Q2R|MATDYN|DYNMAT)|NEB/Doc/INPUT_NEB)\.def",
           r"(?:PW|PHonon|PP|NEB)/examples/[^/]+/(?:README(?:\.md)?|run_example)", r"EPW/examples/(?:[^/]+/)+[^/]+\.in"),
}


def _get(url):
    request = urllib.request.Request(url, headers={"User-Agent": "DFT-Agent"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def pin(engine, commit):
    """Path, SHA-256 and size of every documentation file at a commit."""
    entry = json.loads(CATALOG.read_text(encoding="utf-8"))[engine]
    repository = entry["repository"].removeprefix("https://github.com/")
    meta = json.loads(_get(f"https://api.github.com/repos/{repository}/commits/{commit}"))
    tree = json.loads(_get(f"https://api.github.com/repos/{repository}/git/trees/{meta['commit']['tree']['sha']}?recursive=1"))
    if tree.get("truncated"):
        raise SystemExit("GitHub truncated the file list.")
    paths = sorted(item["path"] for item in tree["tree"]
                   if item["type"] == "blob" and any(re.fullmatch(pattern, item["path"]) for pattern in PATTERNS[engine]))
    raw = entry["raw"].format(commit=meta["sha"])

    def one(path):
        data = _get(raw + path)
        return {"path": path, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
    with ThreadPoolExecutor(8) as pool:
        return meta["sha"], list(pool.map(one, paths))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("engine", choices=sorted(PATTERNS))
    parser.add_argument("version")
    parser.add_argument("commit")
    parser.add_argument("--default", action="store_true", help="use this release when an environment's version is unknown")
    arguments = parser.parse_args()
    commit, pinned = pin(arguments.engine, arguments.commit)
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    entry = catalog[arguments.engine]
    entry["releases"][arguments.version] = {"commit": commit, "files": pinned}
    entry["releases"] = dict(sorted(entry["releases"].items(), key=lambda item: tuple(map(int, item[0].split(".")))))
    if arguments.default:
        entry["default"] = arguments.version
    text = json.dumps(catalog, indent=1, ensure_ascii=False) + "\n"
    # One line per pinned file keeps hundreds of pins readable.
    text = re.sub(r'\{\n\s*("path": "[^"\n]*"),\n\s*("sha256": "[0-9a-f]{64}"),\n\s*("size": \d+)\n\s*\}', r"{\1, \2, \3}", text)
    CATALOG.write_text(text, encoding="utf-8")
    print(f"{arguments.engine} {arguments.version}: {len(pinned)} files, {sum(item['size'] for item in pinned) / 1e6:.1f} MB at {commit}")


if __name__ == "__main__":
    main()
