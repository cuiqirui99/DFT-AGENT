"""Official manuals and examples of QE, Elk, ABINIT and CP2K, built on this computer and searched locally.

DFT Agent ships only where these texts come from (references.json): pinned release archives and
files with their SHA-256, and a few lessons from its own cluster runs. The manuals and examples
(GPL) are downloaded on first use; CP2K's keyword reference is written by the installed program
(cp2k --xml) on the cluster. Texts are kept under the DFT Agent home and ranked locally;
selected excerpts are sent to the user's configured model with the planning request.
"""
from __future__ import annotations

import ast
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
import gzip
import hashlib
from importlib.resources import files
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import tarfile
import tempfile
import time
import warnings
import xml.etree.ElementTree as ET
from uuid import uuid4

from .paths import app_home

ENGINES = ("elk", "abinit", "cp2k", "qe")
LABELS = {"elk": "Elk", "abinit": "ABINIT", "cp2k": "CP2K", "qe": "QE"}
KINDS = ("keyword", "section", "task", "guide", "example", "lesson")
MAX_MEMBER_BYTES = 4_000_000
MAX_XML_BYTES = 200_000_000
MAX_TEXT = 12_000
MAX_READ = 16_000
MAX_RESULTS = 20
RETRY_SECONDS = 3600
QUICK_TIMEOUT = 30
_CP2K_PROGRAMS = ("cp2k.psmp", "cp2k.popt", "cp2k.ssmp", "cp2k.sopt", "cp2k")
_ENGINE_VERSION = re.compile(r"(?i)(?<![A-Za-z0-9])(elk|abinit|cp2k|qe|quantumespresso)[/-]v?(\d+(?:\.\d+)+)")


@lru_cache(maxsize=1)
def catalog():
    return json.loads(files("vasp_slurm_agent").joinpath("references.json").read_text(encoding="utf-8"))


def root():
    return app_home() / "references"


def _downloads(name, cache_dir=None):
    # Without DFT_AGENT_HOME this is where the Elk species installer keeps the same release archive.
    return Path(cache_dir).expanduser() if cache_dir else app_home() / "downloads" / name


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, separators=(",", ":"))
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return path


def _fetch(item, cache, timeout):
    from .engine_setup import _download
    return _download(item, cache, timeout=timeout)


def _fetch_many(items, cache, timeout):
    # One budget for the batch, not a fresh timeout for every file in the queue.
    deadline = time.monotonic() + timeout

    def fetch(item):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("The reference download timed out; cached files can be reused on retry.")
        return _fetch(item, cache, remaining)

    with ThreadPoolExecutor(6) as pool:
        return list(pool.map(fetch, items))


def _version_key(text):
    return tuple(int(part) for part in re.findall(r"\d+", str(text)))


# --- environments --------------------------------------------------------------

def _fingerprint(config):
    identity = [getattr(config, key, None) for key in ("host", "user", "engine_command", "qe_command")]
    identity.append(list(getattr(config, "setup_commands", None) or []))
    if getattr(config, "qe_bin_dir", None):
        identity.append(config.qe_bin_dir)
    return hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:24]


def _bindings():
    try:
        value = json.loads((root() / "environments.json").read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _bind(config, engine, version):
    value = _bindings()
    value[_fingerprint(config)] = {"engine": engine, "version": version, "checked_at": _now()}
    _write(root() / "environments.json", value)


def environment_version(engine, config):
    """The version an environment runs: from its module or managed folder name, or an earlier check."""
    if config is None:
        return None
    text = " ".join([*(getattr(config, "setup_commands", None) or []), getattr(config, "engine_command", None) or "",
                     getattr(config, "qe_bin_dir", None) or ""])
    for name, version in _ENGINE_VERSION.findall(text):
        if {"quantumespresso": "qe"}.get(name.lower(), name.lower()) == engine:
            return version
    value = _bindings().get(_fingerprint(config))
    return value.get("version") if isinstance(value, dict) and value.get("engine") == engine else None


def _program(config, names):
    for token in shlex.split((config.engine_command or "").replace("{tasks}", "1")):
        if PurePosixPath(token).name in names:
            return token
    raise ValueError(f"The environment's command does not name {' or '.join(names)}.")


def release(engine, version):
    """The pinned Elk, ABINIT or QE release whose manual fits a version best: the newest not newer than it."""
    releases = catalog()[engine]["releases"]
    if version in releases:
        return version
    if not version:
        return catalog()[engine]["default"]
    older = [name for name in releases if _version_key(name) <= _version_key(version)]
    return max(older, key=_version_key) if older else min(releases, key=_version_key)


# --- text helpers --------------------------------------------------------------

_GREEK = {name: chr(code) for name, code in (
    ("alpha", 945), ("beta", 946), ("gamma", 947), ("delta", 948), ("epsilon", 949), ("varepsilon", 949), ("zeta", 950),
    ("eta", 951), ("theta", 952), ("kappa", 954), ("lambda", 955), ("mu", 956), ("nu", 957), ("xi", 958), ("pi", 960),
    ("rho", 961), ("sigma", 963), ("tau", 964), ("phi", 966), ("varphi", 966), ("chi", 967), ("psi", 968), ("omega", 969),
    ("Gamma", 915), ("Delta", 916), ("Theta", 920), ("Lambda", 923), ("Sigma", 931), ("Phi", 934), ("Psi", 936), ("Omega", 937))}
_SYMBOLS = {"times": "×", "cdot": "·", "le": "≤", "ge": "≥", "leq": "≤", "geq": "≥", "rightarrow": "→", "to": "→",
            "infty": "∞", "sum": "Σ", "int": "∫", "nabla": "∇", "partial": "∂", "approx": "≈", "sim": "~", "equiv": "≡",
            "pm": "±", "hbar": "ħ", "langle": "⟨", "rangle": "⟩", "uparrow": "↑", "downarrow": "↓", "ne": "≠", "neq": "≠",
            "propto": "∝", "AA": "Å", "dagger": "†", "ldots": "...", "dots": "...", "cdots": "..."}
_ACCENTS = {('"', "o"): "ö", ('"', "a"): "ä", ('"', "u"): "ü", ('"', "O"): "Ö", ("'", "e"): "é", ("'", "a"): "á",
            ("`", "e"): "è", ("^", "o"): "ô", ("~", "n"): "ñ"}


def _cells(row):
    return [cell.strip() for cell in re.split(r"(?<!\\)&", row)]


def _table(body):
    rows = []
    for row in re.split(r"\\\\", body):
        cells = [" ".join(cell.split()) for cell in _cells(row.replace("\\hline", "")) if cell.strip()]
        if cells:
            rows.append("\x05" + cells[0] + (": " + " | ".join(cells[1:]) if len(cells) > 1 else ""))
    return "\n\n" + "\n".join(rows) + "\n\n"


def _latex(text):
    """Readable plain text from the LaTeX in Elk's manual."""
    kept = []

    def keep(match):
        kept.append("\n".join("\x06" + line for line in match.group(1).strip("\n").splitlines()))
        return f"\n\n\x00{len(kept) - 1}\x00\n\n"
    text = re.sub(r"\\begin\{verbatim\}(.*?)\\end\{verbatim\}", keep, text, flags=re.S)
    text = text.replace("\\{", "\x01").replace("\\}", "\x02").replace("\\&", "\x03").replace("\\$", "\x04")
    text = re.sub(r"\\begin\{tabularx?\}(?:\{[^}]*\})?(?:\[[^]]*\])?\{[^}]*\}(.*?)\\end\{tabularx?\}",
                  lambda match: _table(match.group(1)), text, flags=re.S)
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\{\\(?:tt|bf|it|rm|em|sf|sc|cal)\s*([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\(?:textbf|textit|texttt|textrm|emph|mathrm|mathbf|mathcal|mathbb|mbox|text|hat|tilde|widetilde|"
                      r"vec|bar|overline|underline|overleftrightarrow|boldsymbol|operatorname)\s*\{([^{}]*)\}", r"\1", text)
        text = re.sub(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}", r"(\1)/(\2)", text)
        text = re.sub(r"\\sqrt\s*\{([^{}]*)\}", r"√(\1)", text)
    text = re.sub(r"\\([\"'`^~])\{?([A-Za-z])\}?", lambda m: _ACCENTS.get((m.group(1), m.group(2)), m.group(2)), text)
    text = re.sub(r"\\(aa|o|ss)(?![A-Za-z])", lambda m: {"aa": "å", "o": "ø", "ss": "ß"}[m.group(1)], text)
    text = re.sub(r"\\(?:begin|end)\{[^}]*\}(?:\{[^}]*\}|\[[^]]*\])*", "", text)
    text = re.sub(r"\\(?:vskip|hskip|vspace|hspace)\s*\{?-?[\d.]+\s*(?:pt|cm|mm|em|ex)\}?", "\n", text)
    text = re.sub(r"\\label\{[^}]*\}|\\(?:left|right|big|Big|bigg|Bigg|displaystyle|hline|noindent|newline|item)(?![A-Za-z])", "", text)
    text = text.replace("\\\\", "\n")
    text = re.sub(r"\\([%_#])", r"\1", text)
    text = re.sub(r"\\(?:[,;:!]|quad|qquad)", " ", text)
    text = re.sub(r"\\([A-Za-z]+)", lambda m: _GREEK.get(m.group(1)) or _SYMBOLS.get(m.group(1)) or m.group(1), text)
    text = text.replace("$$", "\n").replace("$", "").replace("~", " ").replace("{", "").replace("}", "")
    text = text.replace("``", '"').replace("''", '"').replace("--", "–")
    text = text.replace("\x01", "{").replace("\x02", "}").replace("\x03", "&").replace("\x04", "$")
    text = re.sub(r"\x00(\d+)\x00", lambda m: kept[int(m.group(1))], text)
    paragraphs = []
    for paragraph in re.split(r"\n\s*\n", text):
        merged = []
        for line in paragraph.split("\n"):
            if line.startswith("\x06"):
                merged.append(line.rstrip())
            elif line.strip() and (not merged or line.startswith("\x05") or merged[-1].startswith("\x06")):
                merged.append(" ".join(line.split()))
            elif line.strip():
                merged[-1] += " " + " ".join(line.split())
        if merged:
            paragraphs.append("\n".join(merged))
    return "\n\n".join(paragraphs).replace("\x05", "").replace("\x06", "").strip()


def _first_sentence(text, limit=300):
    text = " ".join(str(text or "").split())
    match = re.match(r"(.{20,}?[.!?])(?:\s|$)", text)
    return (match.group(1) if match else text)[:limit]


def _cp2k_links(text):
    """Markdown links into CP2K's input reference become section paths."""
    def link(match):
        label, target = match.group(1), match.group(2)
        if target.startswith("#CP2K_INPUT."):
            path = target[len("#CP2K_INPUT."):].replace(".", "/")
            return label if label.strip("`").upper() == path.upper() else f"{label} ({path})"
        return label
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", link, text)


# --- Elk -----------------------------------------------------------------------

def _closing(text, start):
    depth, index = 1, start
    while index < len(text):
        character = text[index]
        if character == "\\":
            index += 2
            continue
        if character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return index
        index += 1
    return -1


def _elk_fields(table):
    fields = []
    for row in re.split(r"\\\\|\\hline", table):
        cells = _cells(row)
        if len(cells) < 4 or not cells[0].strip():
            continue
        fields.append({"name": _latex(cells[0]), "description": _latex(" & ".join(cells[1:-2])),
                       "type": _latex(cells[-2]), "default": _latex(cells[-1])})
    return fields


def _task_numbers(label):
    parts = label.split("/")
    numbers = [int(parts[0])]
    for part in parts[1:]:
        numbers.append(int(parts[0][:len(parts[0]) - len(part)] + part))
    return numbers


def _elk_tasks(body):
    records = []
    for table in re.findall(r"\\begin\{tabularx\}\{[^}]*\}(?:\[[^]]*\])?\{[^}]*\}(.*?)\\end\{tabularx\}", body, flags=re.S):
        for row in re.split(r"\\\\", table):
            cells = _cells(row)
            if len(cells) < 2 or not re.fullmatch(r"\d+(?:/\d+)*", cells[0]):
                continue
            label, text = cells[0], _latex(" & ".join(cells[1:]))
            numbers = _task_numbers(label)
            records.append({"id": "task:" + label, "kind": "task", "name": label, "title": "Task " + label,
                            "aliases": [str(number) for number in numbers] + [f"task {number}" for number in numbers],
                            "numbers": numbers, "summary": _first_sentence(text), "text": text})
    return records


def parse_elk_manual(source):
    """Input blocks, tasks and the general sections documented in elk.f90's LaTeX comments."""
    lines = source.splitlines()
    first = next((index for index, line in enumerate(lines) if line.strip() == "!BOI"), -1) + 1
    last = next((index for index in range(len(lines) - 1, -1, -1) if lines[index].strip() == "!EOI"), len(lines))
    body = [re.sub(r"^! {0,3}", "", line) if line.startswith("!") else "" for line in lines[first:last]]
    text = "\n".join(body)

    def line_of(offset):
        return first + 1 + text.count("\n", 0, offset)
    start = text.find("\\section{Input blocks}")
    if start < 0:
        raise ValueError("elk.f90 has no input-block documentation.")
    stop = text.find("\\section{", start + 1)
    stop = len(text) if stop < 0 else stop
    records = []
    heading = re.compile(r"\\(?:sub)*section\{([^}]*)\}")
    sections = list(heading.finditer(text, 0, start))
    for index, match in enumerate(sections):
        title = match.group(1)
        end = sections[index + 1].start() if index + 1 < len(sections) else start
        content = _latex(text[match.end():end])
        if title in {"Acknowledgments", "Licensing"} or len(content) < 80:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        records.append({"id": "guide:" + slug, "kind": "guide", "name": title, "title": "Elk manual: " + title,
                        "summary": _first_sentence(content), "text": content[:MAX_TEXT], "source": f"src/elk.f90:{line_of(match.start())}"})
    blocks = list(re.finditer(r"\\block\{([^}]*)\}\{", text[start:stop]))
    for index, match in enumerate(blocks):
        begin = start + match.end()
        close = _closing(text, begin)
        if close < 0:
            continue
        end = start + blocks[index + 1].start() if index + 1 < len(blocks) else stop
        name = match.group(1).replace("\\_", "_")
        fields = _elk_fields(text[begin:close])
        raw = text[close + 1:end]
        description = _latex(raw)
        heading_text = next((field["description"] for field in fields if re.sub(r"\(.*", "", field["name"]).strip() == name), "")
        record = {"id": "keyword:" + name, "kind": "keyword", "name": name, "title": f"{name}: {heading_text}" if heading_text else name,
                  "aliases": sorted({re.sub(r"\(.*", "", field["name"]).strip() for field in fields} - {name}),
                  "fields": fields, "summary": _first_sentence(description) or heading_text, "text": description[:MAX_TEXT],
                  "source": f"src/elk.f90:{line_of(start + match.start())}"}
        if len(fields) == 1:
            record.update(type=fields[0]["type"], default=fields[0]["default"])
        records.append(record)
        if name == "tasks":
            records.extend({**task, "source": record["source"]} for task in _elk_tasks(raw))
    return records


_SYMBOL_NAMES = None


def _element(symbol):
    global _SYMBOL_NAMES
    if _SYMBOL_NAMES is None:
        from pymatgen.core.periodic_table import Element
        _SYMBOL_NAMES = {element.symbol for element in Element}
    return symbol if symbol in _SYMBOL_NAMES else None


def parse_elk_example(relative, source):
    """One elk.in from the examples folder: its own description, blocks, tasks and species."""
    parts = PurePosixPath(relative).parts
    folder = "/".join(parts[1:-1]) if parts and parts[0] == "examples" else "/".join(parts[:-1])
    comments, blocks, tasks, current, started = [], [], [], None, False
    for line in source.splitlines():
        stripped = line.strip()
        if not started and (stripped.startswith("!") or not stripped):
            comments.append(stripped.lstrip("!").strip())
            continue
        started = True
        if not stripped:
            current = None
            continue
        if stripped.startswith("!"):
            continue
        match = re.fullmatch(r"([a-z][a-z0-9_]*)\s*(?:[!:].*)?", stripped)
        if match and not line[:1].isspace():
            current = match.group(1)
            blocks.append(current)
        elif current == "tasks":
            tasks += [int(value) for value in re.findall(r"^-?\d+", stripped)]
    summary = " ".join(" ".join(comments).split())
    elements = sorted({symbol for symbol in re.findall(r"'([A-Z][a-z]?)\.in'", source) if _element(symbol)})
    name = parts[-2] if len(parts) > 1 else relative
    category = folder.split("/")[0] if "/" in folder else ""
    return {"id": "example:" + folder, "kind": "example", "name": name, "title": folder,
            "summary": summary[:600], "text": source[:MAX_TEXT], "blocks": list(dict.fromkeys(blocks)),
            "tasks": tasks, "elements": elements, "tags": [category] if category else [], "source": relative}


def _elk_records(archive, top):
    manual, examples = None, []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = member.name
            if not member.isfile() or member.size > MAX_MEMBER_BYTES or not name.startswith(top + "/"):
                continue
            relative = name[len(top) + 1:]
            if relative == "src/elk.f90":
                manual = tar.extractfile(member).read().decode("utf-8", "replace")
            elif relative.startswith("examples/") and relative.endswith("/elk.in"):
                examples.append(parse_elk_example(relative, tar.extractfile(member).read().decode("utf-8", "replace")))
    if manual is None:
        raise ValueError("The Elk archive has no src/elk.f90.")
    return parse_elk_manual(manual) + sorted(examples, key=lambda record: record["id"])


def build_elk(version=None, *, cache_dir=None, timeout=120):
    """Download the pinned Elk release, check its hash, and index its manual and examples."""
    version = release("elk", version)
    pinned = catalog()["elk"]["releases"][version]
    archive = _fetch({key: pinned[key] for key in ("url", "sha256", "size")} | {"name": pinned["filename"]},
                     _downloads("elk", cache_dir), timeout)
    records = _elk_records(archive, pinned["root"])
    payload = {"schema_version": 1, "engine": "elk", "part": "manual", "version": version, "built_at": _now(),
               "sources": [{"url": pinned["url"], "sha256": pinned["sha256"]}], "license": catalog()["elk"]["license"],
               "records": records}
    _write(root() / f"elk-{version}.json", payload)
    return {"engine": "elk", "version": version, "records": len(records),
            "examples": sum(record["kind"] == "example" for record in records)}


# --- ABINIT --------------------------------------------------------------------

_ABINIT_UNITS = {"ENERGY": "Hartree, or eV, meV, Ry or K written after the value",
                 "LENGTH": "Bohr, or Angstrom or nm written after the value",
                 "MAGNETIC_FIELD": "atomic units, or T written after the value",
                 "TIME": "atomic units, or s, as or ps written after the value"}
# Plain words that are also ABINIT variable names: a user who writes them means the word.
_ABINIT_WORDS = frozenset("accuracy charge friction positron structure vacuum vis".split())
# What an example does that DFT Agent's ABINIT does not, by the example's tags.
_ABINIT_OUTSIDE = {"paw": "PAW", "dft+u": "DFT+U", "hybrid": "hybrid functionals", "gw": "GW", "bse": "Bethe-Salpeter",
                   "eph": "electron-phonon", "dmft": "DMFT", "rttddft": "real-time TDDFT", "md": "molecular dynamics",
                   "elastic": "strain response", "berry phase": "Berry-phase fields", "nonlinear": "nonlinear response",
                   "wannier": "Wannier functions", "vdw": "van der Waals corrections", "positron": "positrons"}
_ABINIT_DIGEST = {"abinit": ("ndtset", "optdriver", "ecut", "pawecutdg", "ngkpt", "kptopt", "nband", "ixc", "occopt", "tsmear",
                             "nsppol", "nspinor", "rfphon", "rfelfd", "rfstrs", "ngqpt", "ionmov", "optcell", "usepawu"),
                  "anaddb": ("ifcflag", "ngqpt", "dipdip", "asr", "chneut", "nph1l", "prtdos", "thmflag", "dieflag", "elaflag",
                             "piezoflag", "elphflag")}


def _camel(text):
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", text).replace("_", " ")


def _abinit_links(text):
    """ABINIT's wiki links ([[ecut]], [[asr@anaddb]], [[cite:Gonze1997]], [[tutorial:rf1|label]]) as plain words."""
    def link(match):
        target, _, label = match.group(1).partition("|")
        target = target.strip()
        if label.strip():
            return " ".join(label.split())
        prefix, _, name = target.partition(":")
        if not name:
            base, _, program = target.partition("@")
            return f"{base} ({program})" if program and program != "abinit" else base
        if prefix in {"tutorial", "lesson", "topic", "test", "theory"}:
            return f"{'tutorial' if prefix == 'lesson' else prefix} {name}"
        if prefix in {"help", "varset"}:
            return f"{name} {'guide' if prefix == 'help' else 'variables'}"
        return name if prefix in {"abinit", "cite"} or "/" in prefix else f"{name} ({prefix})"
    text = re.sub(r"\{%\s*dialog\s+([^%]+?)\s*%\}", lambda m: "(Input: " + ", ".join(m.group(1).split()) + ")", text)
    text = re.sub(r"\[\[([^\[\]]{1,300}?)\]\]", link, text)
    # Links the source itself left unbalanced, such as [[occopt] or [nband]].
    text = re.sub(r"\[\[([a-z][a-z0-9_@:]{2,})\](?!\])|(?<!\[)\[([a-z][a-z0-9_@]{2,})\]\]", lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r"!\[[^\]\n]*\]\([^)\s]*\)", "", text)
    return re.sub(r"\[([^\]\n]+)\]\((?:[^)\s]+)\)", r"\1", text)


def _abinit_value(node):
    """A default or dimension in ABINIT's variable database as text, read without running its code."""
    if isinstance(node, ast.Call):
        kind = getattr(node.func, "id", "")
        arguments = {keyword.arg: _abinit_value(keyword.value) for keyword in node.keywords}
        if kind == "ValueWithUnit":
            return f"{arguments.get('value', '')} {arguments.get('units', '')}".strip()
        if kind == "MultipleValue":
            number = arguments.get("number")
            return f"{'' if number in (None, 'None') else number}*{arguments.get('value', '')}"
        if kind in {"ValueWithConditions", "Range"} and node.args and isinstance(node.args[0], ast.Dict):
            pairs = [(_abinit_value(key), _abinit_value(value)) for key, value in zip(node.args[0].keys, node.args[0].values)]
            if kind == "Range":
                return " to ".join(value for _, value in pairs)
            return "; ".join([f"{value} if {key}" for key, value in pairs if key != "defaultval"]
                             + [f"otherwise {value}" for key, value in pairs if key == "defaultval"])
        return ast.unparse(node)
    if isinstance(node, (ast.List, ast.Tuple)):
        return "(" + ", ".join(_abinit_value(item) for item in node.elts) + ")"
    try:
        return str(ast.literal_eval(node))
    except ValueError:
        return ast.unparse(node)


def parse_abinit_variables(source, program="abinit"):
    """The input variables in one of ABINIT's abimkdocs/variables_*.py files, parsed as data."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        tree = ast.parse(source)
    records = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", "") == "Variable"):
            continue
        fields = {keyword.arg: keyword.value for keyword in node.keywords}
        try:
            full = ast.literal_eval(fields["abivarname"])
            text = ast.literal_eval(fields.get("text", ast.Constant("")))
            topics = list(ast.literal_eval(fields.get("topics", ast.List([]))))
            characteristics = [item.strip("[]") for item in ast.literal_eval(fields.get("characteristics", ast.List([])))]
        except (KeyError, ValueError):
            continue
        name = full.split("@")[0]
        full = full if program == "abinit" or "@" in full else f"{full}@{program}"
        text = _abinit_links(text).strip()
        mnemonics = _abinit_links(_abinit_value(fields["mnemonics"])) if "mnemonics" in fields else ""
        record = {"id": "keyword:" + full, "kind": "keyword", "name": name, "program": program,
                  "title": f"{name}{' (' + program + ')' if program != 'abinit' else ''}: {mnemonics}".rstrip(": "),
                  "aliases": [full] if full != name else [], "summary": _first_sentence(text), "text": text[:MAX_TEXT],
                  "tags": sorted({_camel(topic.rsplit("_", 1)[0].strip()) for topic in topics}), "topics": topics,
                  "source": f"abimkdocs/variables_{program}.py"}
        for key in ("vartype", "defaultval", "dimensions", "varset", "requires", "excludes", "commentdefault", "added_in_version"):
            if key in fields:
                value = _abinit_links(_abinit_value(fields[key])).strip()
                if value and value != "None":
                    record[{"vartype": "type", "defaultval": "default"}.get(key, key)] = value
        if characteristics:
            record["characteristics"] = characteristics
        units = [_ABINIT_UNITS[item] for item in characteristics if item in _ABINIT_UNITS]
        if units:
            record["unit"] = units[0]
        # Basic variables first; developer and internal ones only when nothing else matches.
        record["prior"] = 1.5 if record.get("varset") == "basic" else (
            -4.0 if record.get("varset") in {"dev", "internal"} or {"DEVELOP", "INTERNAL_ONLY"} & set(characteristics) else 0.0)
        records.append(record)
    return records


def _front_matter(text):
    match = re.match(r"---\n(.*?)\n---\n", text, flags=re.S)
    if not match:
        return {}, text
    fields = dict(re.findall(r"(?m)^(\w+):\s*(.*)$", match.group(1)))
    return fields, text[match.end():]


def parse_abinit_markdown(relative, source, related=None):
    """A tutorial, topic or guide page of ABINIT's documentation, split at its second-level headings."""
    fields, text = _front_matter(source)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    if related is not None:
        text = text.replace("{{ related_variables }}", "\n".join(f"{level.capitalize()}: {', '.join(names[:40])}"
                                                                 for level, names in related.items()))
    text = _abinit_links(text)
    text = re.sub(r"\{%.*?%\}|\{\{.*?\}\}|^\[TUTORIAL_README\w*\]$", "", text, flags=re.S | re.M)
    text = re.sub(r"(?m)^([!?]{3})[ \t]*(\w+)(?:[ \t]+\"([^\"]*)\")?[ \t]*$",
                  lambda m: (m.group(3) or m.group(2).capitalize()) + ":", text)
    parts = PurePosixPath(relative)
    if parts.parent.name == "topics":
        topic = parts.stem.lstrip("_")
        page, tags = f"Topic {_camel(topic)}", ["topic", _camel(topic)]
    else:
        title_match = re.search(r"^#\s+(.+)$", text, flags=re.M)
        page = title_match.group(1).strip().rstrip(":") if title_match else parts.stem
        tags = [{"tutorial": "tutorial", "guide": "guide"}.get(parts.parent.name, parts.parent.name), parts.stem]
    return _markdown_records(relative, text, page, tags, summary=_abinit_links(fields.get("description", "")) or None)


def _abinit_tokens(line):
    return re.findall(r'"[^"]*"|\'[^\']*\'|\S+', re.split(r"[#!]", line, maxsplit=1)[0])


def parse_abinit_input(relative, source, names):
    """One input of ABINIT's test suite: its description, test keywords, settings, elements and features.

    names maps each program (abinit, anaddb) to its variable names, so that dataset suffixes such as
    getwfk2 or ecut: are read as getwfk and ecut and other words are ignored."""
    info, body = {}, []
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#%%"):
            match = re.match(r"(\w+)\s*=\s*(.*)$", stripped[3:].strip())
            if match and match.group(2).strip():
                info.setdefault(match.group(1), match.group(2).strip())
            continue
        body.append(line)
    text = "\n".join(body).rstrip()
    text = re.sub(r"\n#{20,}\n# This section is used only for regression testing of ABINIT #\n#{20,}\s*$", "", text).rstrip()
    program = info.get("executable", "abinit").split()[0]
    known = names.get(program, frozenset())
    comments, values, order, current = [], defaultdict(list), [], None
    for line in text.splitlines():
        stripped = line.strip()
        if not order and not current and (stripped.startswith(("#", "!")) or not stripped):
            comments.append(stripped.lstrip("#!").strip())
            continue
        for token in _abinit_tokens(line):
            base = re.sub(r"[?:+*]$", "", token.lower())
            stem = re.sub(r"\d+$", "", base)
            name = base if base in known else stem if stem in known else None
            if name and not token[:1].isdigit():
                current = [name, base[len(name):], []]
                values[name].append(current)
                if name not in order:
                    order.append(name)
            elif current is not None:
                current[2].append(token)

    def first(name):
        return " ".join(values[name][0][2])[:60] if values.get(name) else None

    def numbers(name):
        return [int(match) for _, _, tokens in values.get(name, []) for match in re.findall(r"^[-+]?\d+$", tokens[0] if tokens else "")]
    elements = []
    if values.get("znucl"):
        from pymatgen.core.periodic_table import Element
        for token in values["znucl"][0][2]:
            try:
                elements.append(Element.from_Z(int(float(token))).symbol)
            except (ValueError, KeyError):
                break
    pseudos = " ".join(values["pseudos"][0][2]) if values.get("pseudos") else ""
    ionmov, optdriver, ixc = set(numbers("ionmov")), set(numbers("optdriver")), {str(value) for value in numbers("ixc")}
    features = {"phonons": any(numbers("rfphon")), "dielectric": any(numbers("rfelfd")), "elastic": any(numbers("rfstrs")),
                "relaxation": bool(ionmov & {2, 3, 22}) or any(numbers("optcell")), "md": bool(ionmov & {6, 8, 9, 12, 13}),
                "bands": any(value < 0 for value in numbers("kptopt")), "dos": any(numbers("prtdos")),
                "spin": 2 in numbers("nsppol"), "noncollinear": 2 in numbers("nspinor"), "dft+u": any(numbers("usepawu")),
                "paw": "pawecutdg" in values or bool(re.search(r"(?i)\.xml\b|\.paw|jth", pseudos)),
                "gw": bool(optdriver & {3, 4, 6}), "bse": 99 in optdriver, "rttddft": 9 in optdriver,
                "eph": 7 in optdriver or any(numbers("elphflag")) or any(numbers("prtgkk")) or any(numbers("ieig2rf")),
                "nonlinear": 5 in optdriver, "dmft": any(numbers("usedmft")), "berry phase": any(numbers("berryopt")),
                "hybrid": bool(ixc & {"40", "41", "42", "-406", "-427", "-428"}), "wannier": any(numbers("prtwant")),
                "vdw": any(numbers("vdw_xc")), "interatomic force constants": program == "anaddb" and any(numbers("ifcflag")),
                "phonon bands": program == "anaddb" and (any(numbers("nph1l")) or any(numbers("nqpath"))),
                "phonon dos": program == "anaddb" and any(numbers("prtdos")), "thermodynamics": any(numbers("thmflag"))}
    keywords = [word.strip().lower() for word in info.get("keywords", "").split(",") if word.strip()]
    tags = sorted({*keywords, program, *(name for name, on in features.items() if on)})
    digest = []
    for name in _ABINIT_DIGEST.get(program, ()):
        for _, suffix, tokens in values.get(name, [])[:2]:
            digest.append(f"{name}{suffix} {' '.join(tokens)[:40]}".strip())
    natom = first("natom")
    if natom:
        digest.append(f"{natom} atoms" + (f" ({', '.join(sorted(set(elements)))})" if elements else ""))
    header = " ".join(" ".join(comments).split())[:300]
    description = " ".join(info.get("description", "").split())[:200]
    summary = "; ".join(filter(None, [header, "" if description.lower() in header.lower() else description, ", ".join(digest)]))
    return {"id": "example:" + relative, "kind": "example", "name": PurePosixPath(relative).name, "title": relative,
            "program": program, "summary": summary[:700], "text": text[:MAX_TEXT], "elements": sorted(set(elements)),
            "variables": order[:80], "tags": tags, "chain": info.get("test_chain", ""), "source": relative}


def abinit_records(sources):
    """Variables, tutorials, topics, guides and tutorial inputs of one ABINIT release, keyed by path."""
    records, names, related = [], {}, defaultdict(lambda: defaultdict(list))
    for program in ("abinit", "anaddb"):
        found = parse_abinit_variables(sources.get(f"abimkdocs/variables_{program}.py", ""), program)
        names[program] = frozenset(record["name"] for record in found)
        for record in found:
            for topic in record["topics"]:
                base, _, level = topic.rpartition("_")
                related[base.strip()][level].append(record["name"] + ("" if program == "abinit" else f" ({program})"))
        records += found
    order, chains = ("compulsory", "basic", "useful", "expert"), defaultdict(set)
    for path, source in sorted(sources.items()):
        if path.startswith("doc/"):
            topic = PurePosixPath(path).stem.lstrip("_") if "/topics/" in path else None
            levels = related.get(topic) or {}
            records += parse_abinit_markdown(path, source, {level: levels[level] for level in order if level in levels}
                                             if topic else None)
        elif path.endswith(".abi"):
            records.append(parse_abinit_input(path, source, names))
            chains[(str(PurePosixPath(path).parent), records[-1]["chain"])].update(records[-1]["elements"])
    # mrgddb and anaddb steps of a tutorial chain work on the material of its abinit steps.
    for record in records:
        if record["kind"] == "example" and not record["elements"] and record["chain"]:
            record["elements"] = sorted(chains[(str(PurePosixPath(record["source"]).parent), record["chain"])])
    return records


def build_abinit(version=None, *, cache_dir=None, timeout=120):
    """Download the pinned files of an ABINIT release, check their hashes, and index them."""
    version = release("abinit", version)
    pinned = catalog()["abinit"]["releases"][version]
    base = catalog()["abinit"]["raw"].format(commit=pinned["commit"])
    cache = _downloads("abinit-docs", cache_dir) / version
    items = [{"name": item["path"].replace("/", "__"), "url": base + item["path"], "sha256": item["sha256"], "size": item["size"]}
             for item in pinned["files"]]
    paths = _fetch_many(items, cache, timeout)
    sources = {item["path"]: path.read_text(encoding="utf-8", errors="replace") for item, path in zip(pinned["files"], paths)}
    records = abinit_records(sources)
    payload = {"schema_version": 1, "engine": "abinit", "part": "manual", "version": version, "built_at": _now(),
               "sources": [{"url": base, "files": len(items)}], "license": catalog()["abinit"]["license"], "records": records}
    _write(root() / f"abinit-{version}.json", payload)
    return {"engine": "abinit", "version": version, "records": len(records),
            "examples": sum(record["kind"] == "example" for record in records)}


# --- Quantum ESPRESSO ----------------------------------------------------------

_QE_NAMELISTS = {"INPUTEPW": "epw.x", "INPUTPH": "ph.x", "DOS": "dos.x", "PROJWFC": "projwfc.x", "BANDS": "bands.x",
                 "INPUTPP": "pp.x", "PATH": "neb.x", "INPUTHP": "hp.x", "CONTROL": "pw.x", "SYSTEM": "pw.x"}
# Words that are also QE variable names but mean something else in a request.
_QE_WORDS = frozenset("title prefix lambda report charge verbosity".split())
# What an example does that DFT Agent's QE does not, by the example's tags.
_QE_OUTSIDE = {"md": "molecular dynamics", "berry phase": "Berry-phase polarization", "electric field": "finite electric fields",
               "esm": "the effective screening medium", "rism": "RISM solvation", "raman": "Raman tensors",
               "transport": "carrier transport", "polaron": "polarons"}
_QE_DIGEST = {"pw.x": ("calculation", "ecutwfc", "ecutrho", "occupations", "smearing", "degauss", "nspin", "noncolin", "lspinorb",
                       "input_dft", "nqx1", "lda_plus_u", "vdw_corr", "conv_thr", "nbnd"),
              "ph.x": ("ldisp", "nq1", "nq2", "nq3", "tr2_ph", "epsil", "trans", "electron_phonon", "lraman", "asr"),
              "matdyn.x": ("asr", "dos", "nk1", "q_in_band_form", "la2f"), "q2r.x": ("zasr", "la2f"),
              "dos.x": ("degauss", "deltae"), "projwfc.x": ("degauss", "deltae"), "neb.x": ("num_of_images", "path_thr"),
              "epw.x": ("elph", "wannierize", "nbndsub", "nkf1", "nqf1", "eliashberg", "liso", "laniso", "muc", "degaussw",
                        "scattering", "iterative_bte")}


def _tcl(text):
    """Commands of a Tcl script as lists of words; a braced word is kept as ("{", its raw text)."""
    commands, words, index, size = [], [], 0, len(text)
    while index < size:
        character = text[index]
        if character in "\n;":
            if words:
                commands.append(words)
                words = []
            index += 1
        elif character in " \t\r":
            index += 1
        elif character == "\\" and text[index + 1:index + 2] == "\n":
            index += 2
        elif character == "#" and not words:
            end = text.find("\n", index)
            index = size if end < 0 else end
        elif character == "{":
            depth, end = 1, index + 1
            while end < size and depth:
                if text[end] == "\\":
                    end += 2
                    continue
                depth += {"{": 1, "}": -1}.get(text[end], 0)
                end += 1
            words.append(("{", text[index + 1:end - 1]))
            index = end
        elif character == '"':
            end = index + 1
            while end < size and text[end] != '"':
                end += 2 if text[end] == "\\" else 1
            words.append(text[index + 1:end])
            index = end + 1
        else:
            end = index
            while end < size and text[end] not in " \t\r\n;":
                end += 1
            words.append(text[index:end])
            index = end
    if words:
        commands.append(words)
    return commands


def _braced(words):
    return next((word[1] for word in reversed(words) if isinstance(word, tuple)), None)


def _tcl_flags(words):
    flags, index = {}, 0
    while index < len(words) - 1:
        word = words[index]
        if isinstance(word, str) and word.startswith("-"):
            value = words[index + 1]
            flags[word[1:]] = value[1] if isinstance(value, tuple) else value
            index += 2
        else:
            index += 1
    return flags


def _helpdoc_text(raw):
    """Plain text from helpdoc's markup (@ref, @b, @i, @br), without the source's indentation."""
    text = re.sub(r"@(?:b|i|tt|ref|link|sup|sub|code)\s*\{([^{}]*)\}", r"\1", raw or "")
    text = re.sub(r"@br\b", "\n", text)
    text = re.sub(r"@(?:b|i|tt|ref|link|sup|sub|code)\s+", "", text)
    lines = [line.strip() for line in text.strip("\n").splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _helpdoc_entry(words):
    """Text, choices and facts (default, units, status, ...) of a var, dimension, col or flag command."""
    texts, choices, facts = [], [], {}
    for command in _tcl(_braced(words) or ""):
        content = _braced(command)
        if content is None:
            continue
        if command[0] in {"default", "status", "units", "dimensionality", "see", "enum"}:
            facts[command[0]] = " ".join(_helpdoc_text(content).split())
        elif command[0] == "info":
            texts.append(_helpdoc_text(content))
        elif command[0] == "options":
            for option in _tcl(content):
                body = _braced(option)
                if option[0] == "info" and body is not None:
                    texts.append(_helpdoc_text(body))
                elif option[0] == "opt":
                    choices.append({"name": _tcl_flags(option[1:]).get("val", ""), "description": " ".join(_helpdoc_text(body).split())})
    return "\n\n".join(filter(None, texts)), choices, facts


def parse_qe_helpdoc(relative, source):
    """Variables, namelists and cards of one of QE's INPUT_*.def input descriptions."""
    records, program = [], PurePosixPath(relative).stem.removeprefix("INPUT_").lower() + ".x"

    def variable(words, section, shared="", group_type=""):
        # "alpha_mix(niter)" names an array by its size.
        name, _, size = words[1].partition("(")
        text, choices, facts = _helpdoc_entry(words)
        text = "\n\n".join(filter(None, [text, shared]))
        flags = _tcl_flags(words[2:])
        record = {"id": f"keyword:{program}/{section}/{name}", "kind": "keyword", "name": name, "program": program,
                  "namelist": section, "title": f"{name} ({program} {section})", "type": (flags.get("type") or group_type).lower(),
                  "summary": _first_sentence(text), "text": text[:MAX_TEXT], "source": relative}
        if words[0] in {"dimension", "multidimension"} or "start" in flags:
            record["dimensions"] = f"({flags.get('start', '1')} to {flags.get('end', '')})"
        elif size:
            record["dimensions"] = f"({size}"
        record.update({key: value for key, value in facts.items() if key in {"default", "status", "dimensionality", "see"}})
        if facts.get("units"):
            record["unit"] = facts["units"]
        if choices:
            record["choices"] = choices[:200]
        # pw.x first; the post-processing programs share many names with it.
        record["prior"] = 3.0 if program == "pw.x" else 0.0
        return record

    def walk(commands, section):
        nonlocal program
        for words in commands:
            # QE 7.4.1's INPUT_PW.def opens with "&input_description", which helpdoc accepts.
            head, content = words[0].lstrip("&") if isinstance(words[0], str) else "", _braced(words)
            if head == "input_description":
                program = _tcl_flags(words[1:]).get("program", program)
                walk(_tcl(content or ""), section)
            elif head in {"namelist", "card"} and len(words) > 1:
                name, start = ("&" if head == "namelist" else "") + words[1], len(records)
                inner = _tcl(content or "")
                flag = next((item for item in inner if item[0] == "flag"), None)
                walk(inner, name)
                text, choices, facts = _helpdoc_entry(flag) if flag else ("", [], {})
                info = "\n\n".join(_helpdoc_text(_braced(item)) for item in inner if item[0] == "info" and _braced(item) is not None)
                text = "\n\n".join(filter(None, [info, text, "\n".join(f"{choice['name']}: {choice['description']}"
                                                                       for choice in choices if choice["description"])]))
                record = {"id": f"section:{program}/{name}", "kind": "section", "name": name, "program": program,
                          "title": f"{program} {name}", "summary": _first_sentence(text) or f"The {name} {head} of {program}.",
                          "text": text[:MAX_TEXT], "keywords": sorted({item["name"] for item in records[start:] if item["kind"] == "keyword"}),
                          "source": relative, "prior": 3.0 if program == "pw.x" else 0.0}
                if choices:
                    record["choices"] = [{"name": choice["name"], "description": choice["description"][:400]} for choice in choices]
                if facts.get("default"):
                    record["default"] = facts["default"]
                records.insert(start, record)
            elif head in {"var", "dimension", "multidimension", "col"} and len(words) > 1:
                records.append(variable(words, section))
            elif head in {"vargroup", "dimensiongroup", "colgroup"}:
                inner = _tcl(content or "")
                shared = "\n\n".join(_helpdoc_text(_braced(item)) for item in inner if item[0] == "info" and _braced(item) is not None)
                group_type = _tcl_flags(words[1:]).get("type", "")
                for item in inner:
                    if item[0] in {"var", "dimension", "multidimension", "col"} and len(item) > 1:
                        records.append(variable(item, section, shared, group_type))
            elif head == "intro" and content:
                text = _helpdoc_text(content)
                records.append({"id": f"guide:{relative}#intro", "kind": "guide", "name": "Introduction",
                                "title": f"{program} input: introduction", "summary": _first_sentence(text), "text": text[:MAX_TEXT],
                                "tags": ["input", program], "source": relative})
            elif content and head in {"supercard", "group", "choose", "when", "elsewhen", "otherwise", "syntax", "line", "table",
                                      "rows", "cols", "optional", "conditional", "section", "subsection"}:
                walk(_tcl(content), section)

    walk(_tcl(source), None)
    seen, unique = set(), []
    for record in records:
        if record["id"] not in seen:
            seen.add(record["id"])
            unique.append(record)
    return unique


def _qe_values(text):
    """Namelist assignments of a QE input, lower-case names to their first values."""
    values = {}
    for name, value in re.findall(r"(?im)\b([a-z][a-z0-9_]*)(?:\([^)]*\))?\s*=\s*('[^'\n]*'|\"[^\"\n]*\"|[^,\s/!]+)", text):
        values.setdefault(name.lower(), value.strip("'\""))
    return values


def parse_qe_input(relative, source, description=""):
    """One QE input: its program, a digest of the settings that matter for planning, elements and features."""
    namelists = [name.upper() for name in re.findall(r"(?im)^\s*&(\w+)", source)]
    values = _qe_values(source)
    program = next((_QE_NAMELISTS[name] for name in namelists if name in _QE_NAMELISTS), None)
    if program is None and "INPUT" in namelists:
        program = "matdyn.x" if {"flfrq", "q_in_band_form", "dos", "fldos"} & set(values) else (
            "q2r.x" if "zasr" in values or "flfrc" in values and "fildyn" in values else "dynmat.x")
    if program is None:
        return None
    elements, species = [], re.search(r"(?ims)^\s*ATOMIC_SPECIES[^\n]*\n(.*?)(?=^\s*(?:[A-Z_]{4,}\b|&|/)|\Z)", source)
    for line in (species.group(1).splitlines() if species else []):
        parts = line.split()
        symbol = _symbol(parts[0]) if len(parts) >= 3 and not parts[0].startswith(("!", "#")) else None
        elements += [symbol] if symbol else []
    lower = {key: value.lower() for key, value in values.items()}

    def true(key):
        return lower.get(key, "").strip(".") in {"true", "t"}
    calculation = lower.get("calculation", "scf" if program == "pw.x" else "")
    features = {"relaxation": calculation in {"relax", "vc-relax"}, "md": calculation in {"md", "vc-md"},
                "bands": calculation == "bands" or program == "bands.x", "nscf": calculation == "nscf",
                "dos": program in {"dos.x", "projwfc.x"} or true("dos"), "pdos": program == "projwfc.x",
                "metal": lower.get("occupations") in {"smearing", "tetrahedra", "tetrahedra_opt"}, "spin": lower.get("nspin") == "2",
                "noncollinear": true("noncolin"), "soc": true("lspinorb"),
                "hybrid": any(name in lower.get("input_dft", "") for name in ("hse", "pbe0", "b3lyp", "hf", "gaupbe")) or "nqx1" in values,
                "dft+u": true("lda_plus_u") or bool(re.search(r"(?im)^\s*HUBBARD\b", source)), "vdw": "vdw_corr" in values or true("london"),
                "berry phase": true("lberry") or true("lelfield"), "electric field": true("tefield") or true("lelfield") or true("gate"),
                "esm": lower.get("assume_isolated") == "esm", "rism": true("trism"),
                "isolated": lower.get("assume_isolated", "none") not in {"none", "esm"},
                "phonons": program in {"ph.x", "q2r.x", "matdyn.x", "dynmat.x"}, "dispersion": true("ldisp") or program == "matdyn.x",
                "dielectric": true("epsil"), "raman": true("lraman"),
                "electron-phonon": program == "epw.x" or bool(lower.get("electron_phonon")) or true("la2f"),
                "superconductivity": true("eliashberg") or true("liso") or true("laniso"), "wannier": true("wannierize"),
                "transport": true("scattering") or true("iterative_bte") or true("lifc"), "polaron": true("plrn"), "neb": program == "neb.x"}
    digest = [f"{name} {values[name]}"[:60] for name in _QE_DIGEST.get(program, ()) if name in values]
    kpoints = re.search(r"(?im)^\s*K_POINTS\s*[({]?\s*(\w+)[)}]?[^\n]*\n([^\n]*)", source)
    if kpoints:
        digest.append(f"K_POINTS {kpoints.group(1)}" + (f" {' '.join(kpoints.group(2).split())}" if kpoints.group(1).lower() == "automatic" else ""))
    if elements:
        digest.append(f"{values.get('nat', len(elements))} atoms ({', '.join(sorted(set(elements)))})")
    comments = " ".join(line.strip().lstrip("!#").strip() for line in source.splitlines()[:6] if line.strip().startswith(("!", "#")))
    summary = "; ".join(filter(None, [" ".join(description.split())[:300], comments[:200], ", ".join(digest)]))
    return {"id": "example:" + relative, "kind": "example", "name": PurePosixPath(relative.split("#")[-1]).name, "title": relative,
            "program": program, "summary": summary[:700], "text": source.strip()[:MAX_TEXT], "elements": sorted(set(elements)),
            "variables": list(values)[:80], "tags": sorted({program, *(name for name, on in features.items() if on)}),
            "source": relative.split("#")[0]}


def _readme_step(readme, name):
    """The paragraph of an example's README that describes the input with this file name."""
    stem = re.sub(r"\$\{?\w+\}?|\.in\w*$", "", name).strip(".")
    stem = re.sub(r"\.{2,}", ".", stem)
    for paragraph in re.split(r"\n\s*\n", readme):
        if stem and len(stem) > 2 and stem in paragraph:
            return " ".join(paragraph.split())
    return ""


def parse_qe_example_script(relative, source, readme=""):
    """The QE inputs an official example script writes as here-documents, each with its README step."""
    records, names = [], Counter()
    introduction = " ".join(re.split(r"\n\s*\n", readme.strip())[0].split()) if readme.strip() else ""
    for match in re.finditer(r"(?m)^[ \t]*cat\s*>\s*(\S+)\s*<<\s*['\"]?(\w+)['\"]?[^\n]*\n(.*?)^\2[ \t]*$", source, flags=re.S):
        name, body = match.group(1), match.group(3)
        after = re.search(r'\$ECHO\s+"\s*(running [^"\\]*)', source[match.end():match.end() + 400])
        step = _readme_step(readme, PurePosixPath(name).name) or (after.group(1).strip() if after else "")
        names[name] += 1
        suffix = "" if names[name] == 1 else f"-{names[name]}"
        record = parse_qe_input(f"{relative}#{name}{suffix}", body, "; ".join(filter(None, [introduction[:250], step[:300]])))
        if record:
            records.append(record)
    _inherit_elements(records, lambda record: record["name"].split(".")[0].lower())
    return records


def _inherit_elements(records, key):
    """ph.x, q2r, matdyn and EPW inputs list no species: they take those of the pw.x input of the same
    material (the same key), or of the example when it has only one material."""
    found, materials = defaultdict(set), {tuple(record["elements"]) for record in records if record["elements"]}
    for record in records:
        found[key(record)].update(record["elements"])
    for record in records:
        if not record["elements"]:
            record["elements"] = sorted(found[key(record)] or (set(next(iter(materials))) if len(materials) == 1 else set()))


def qe_records(sources):
    """Input descriptions, example READMEs and example inputs of one QE release, keyed by path."""
    records = []
    for path, source in sorted(sources.items()):
        parts = PurePosixPath(path)
        if path.endswith(".def"):
            records += parse_qe_helpdoc(path, source)
        elif parts.name.startswith("README"):
            folder = str(parts.parent)
            text = source.strip()
            records.append({"id": f"guide:{path}", "kind": "guide", "name": folder, "title": f"QE example {folder}",
                            "summary": _first_sentence(text), "text": text[:MAX_TEXT], "tags": ["example", parts.parts[0]],
                            "source": path})
        elif parts.name == "run_example":
            readme = next((sources[str(parts.parent / name)] for name in ("README", "README.md") if str(parts.parent / name) in sources), "")
            records += parse_qe_example_script(path, source, readme)
        elif path.endswith(".in"):
            record = parse_qe_input(path, source, f"EPW example {parts.parts[2]}" if parts.parts[0] == "EPW" else "")
            if record:
                records.append(record)
    epw = [record for record in records if record["kind"] == "example" and record["source"].startswith("EPW/examples/")]
    _inherit_elements(epw, lambda record: record["source"].split("/")[2])
    return records


# DFT Agent's namelist sections and the program and namelist each one is written to.
_QE_SECTIONS = {"control": ("pw.x", "&CONTROL"), "system": ("pw.x", "&SYSTEM"), "electrons": ("pw.x", "&ELECTRONS"),
                "ions": ("pw.x", "&IONS"), "cell": ("pw.x", "&CELL"), "inputph": ("ph.x", "&INPUTPH"), "dos": ("dos.x", "&DOS"),
                "projwfc": ("projwfc.x", "&PROJWFC"), "q2r": ("q2r.x", "&INPUT"), "matdyn": ("matdyn.x", "&INPUT"),
                "path": ("neb.x", "&PATH")}


def qe_namelist_problems(namelists, config=None):
    """Native QE options whose names the environment's QE version does not document in that namelist.

    QE stops on any name its namelist does not have, but only after the job has waited in the queue.
    A few names QE reads are missing from its descriptions (q2r.x's la2f): names the official examples
    use count as known unless the description puts them in another namelist. Only an index already
    built is used; without one nothing is checked."""
    if not isinstance(namelists, dict) or not namelists:
        return []
    reference = load("qe", config, build=False)
    wanted = environment_version("qe", config)
    if "manual" not in reference.parts or not wanted or reference.parts["manual"]["version"] != wanted:
        return []
    known, used = defaultdict(set), defaultdict(set)
    for record in reference.records:
        if record["kind"] == "keyword" and str(record.get("namelist", "")).startswith("&"):
            known[(record["program"], record["namelist"].upper())].add(record["name"].lower())
        elif record["kind"] == "example":
            used[record.get("program")].update(record.get("variables", []))
    label, problems = f"QE {reference.parts['manual']['version']}", []
    for section, values in namelists.items():
        program, namelist = _QE_SECTIONS.get(str(section).lower(), (None, None))
        if program is None or not isinstance(values, dict) or not known.get((program, namelist)):
            continue
        for key in values:
            name = str(key).split("(")[0].strip().lower()
            if name in known[(program, namelist)]:
                continue
            elsewhere = sorted(other for (owner, other), names in known.items() if owner == program and name in names)
            if not elsewhere and name in used[program]:
                continue
            problems.append(f"{label}'s {program} has no {namelist} variable {name}"
                            + (f"; it belongs to {' or '.join(elsewhere)}." if elsewhere else "."))
    return problems


def build_qe(version=None, *, cache_dir=None, timeout=120):
    """Download the pinned files of a QE release, check their hashes, and index them."""
    version = release("qe", version)
    entry = catalog()["qe"]
    pinned = entry["releases"][version]
    base = entry["raw"].format(commit=pinned["commit"])
    cache = _downloads("qe-docs", cache_dir) / version
    items = [{"name": item["path"].replace("/", "__"), "url": base + item["path"], "sha256": item["sha256"], "size": item["size"]}
             for item in pinned["files"]]
    paths = _fetch_many(items, cache, timeout)
    sources = {item["path"]: path.read_text(encoding="utf-8", errors="replace") for item, path in zip(pinned["files"], paths)}
    records = qe_records(sources)
    payload = {"schema_version": 1, "engine": "qe", "part": "manual", "version": version, "built_at": _now(),
               "sources": [{"url": base, "files": len(items)}], "license": entry["license"], "records": records}
    _write(root() / f"qe-{version}.json", payload)
    return {"engine": "qe", "version": version, "records": len(records),
            "examples": sum(record["kind"] == "example" for record in records)}


# --- CP2K ----------------------------------------------------------------------

_ROOT_RANK = {"GLOBAL": 0, "FORCE_EVAL": 0, "MOTION": 0, "VIBRATIONAL_ANALYSIS": 1, "EXT_RESTART": 1,
              "MULTIPLE_FORCE_EVALS": 1, "ATOM": 3, "TEST": 3, "DEBUG": 3, "OPTIMIZE_INPUT": 3, "OPTIMIZE_BASIS": 3}


def _path_rank(path):
    return (_ROOT_RANK.get(path.split("/", 1)[0], 2), path.count("/"), path)


def _xml_text(element, tag):
    return (element.findtext(tag) or "").strip()


def _cp2k_keyword(element):
    names = element.findall("NAME")
    name = next((item.text for item in names if item.get("type") == "default"), names[0].text if names else "") or ""
    data = element.find("DATA_TYPE")
    choices = [] if data is None else [{"name": _xml_text(item, "NAME"), "description": " ".join(_xml_text(item, "DESCRIPTION").split())}
                                       for item in data.iter("ITEM")]
    return {"name": name.strip(), "aliases": [item.text.strip() for item in names if item.get("type") == "alias" and item.text],
            "type": "" if data is None else data.get("kind", ""), "n_var": "" if data is None else _xml_text(data, "N_VAR"),
            "choices": choices, "usage": _xml_text(element, "USAGE"), "description": _xml_text(element, "DESCRIPTION"),
            "default": _xml_text(element, "DEFAULT_VALUE"), "unit": _xml_text(element, "DEFAULT_UNIT"),
            "lone": _xml_text(element, "LONE_KEYWORD_VALUE"), "repeats": element.get("repeats") == "yes",
            "removed": element.get("removed") == "yes", "deprecated": _xml_text(element, "DEPRECATION_NOTICE"),
            "location": _xml_text(element, "LOCATION")}


def _merge(store, key, path, entry):
    rank = _path_rank(path)
    found = store.get(key)
    if found is None:
        store[key] = {"entry": entry, "path": path, "rank": rank, "count": 1}
    else:
        found["count"] += 1
        if rank < found["rank"]:
            found.update(path=path, rank=rank)


def parse_cp2k_xml(data):
    """Sections and keywords from cp2k --xml, each definition once under its most central path."""
    start = data.find(b"<CP2K_INPUT")
    if start < 0 or b"<!DOCTYPE" in data[:start] or b"<!ENTITY" in data[:start]:
        raise ValueError("This is not CP2K's input reference.")
    sections, keywords, stack, tags, version = {}, {}, [], [], ""
    for event, element in ET.iterparse(io.BytesIO(data), events=("start", "end")):
        tag = element.tag
        if event == "start":
            tags.append(tag)
            if tag == "SECTION":
                stack.append({"name": "", "description": "", "location": "", "keywords": [], "subsections": [],
                              "repeats": element.get("repeats") == "yes"})
            continue
        tags.pop()
        parent = tags[-1] if tags else None
        if tag == "CP2K_VERSION":
            version = (element.text or "").replace("CP2K version", "").strip()
        elif parent == "SECTION" and stack:
            if tag in {"NAME", "DESCRIPTION", "LOCATION"}:
                stack[-1][tag.lower()] = (element.text or "").strip()
            elif tag in {"KEYWORD", "SECTION_PARAMETERS"}:
                entry = _cp2k_keyword(element)
                if tag == "KEYWORD":
                    stack[-1]["keywords"].append(entry)
                else:
                    stack[-1]["parameters"] = entry
                element.clear()
        if tag == "SECTION" and stack:
            section = stack.pop()
            path = "/".join([item["name"] for item in stack] + [section["name"]])
            if stack:
                stack[-1]["subsections"].append(section["name"])
            _merge(sections, (section["location"], section["name"]), path, section)
            for entry in section["keywords"]:
                _merge(keywords, (entry["location"], entry["name"], entry["default"]), path, entry)
            element.clear()
        elif parent == "CP2K_INPUT" and tag in {"REFERENCE", "UNIT_KIND"}:
            element.clear()
    records, seen = [], set()

    def unique(identifier):
        candidate, number = identifier, 2
        while candidate in seen:
            candidate, number = f"{identifier}#{number}", number + 1
        seen.add(candidate)
        return candidate
    for item in sorted(sections.values(), key=lambda value: value["rank"]):
        section, path = item["entry"], item["path"]
        text = _cp2k_links(section["description"])
        record = {"id": unique("section:" + path), "kind": "section", "name": section["name"], "path": path, "title": path,
                  "summary": _first_sentence(text), "text": text[:MAX_TEXT],
                  "keywords": sorted({entry["name"] for entry in section["keywords"]}),
                  "subsections": sorted(set(section["subsections"])), "repeats": section["repeats"], "also_in": item["count"] - 1}
        if "parameters" in section:
            parameters = section["parameters"]
            record["parameters"] = {key: parameters[key] for key in ("type", "default", "lone", "usage") if parameters[key]}
            record["parameters"]["description"] = _cp2k_links(parameters["description"])
        records.append(record)
    for item in sorted(keywords.values(), key=lambda value: value["rank"]):
        entry, path = item["entry"], item["path"]
        text = _cp2k_links(entry["description"])
        record = {"id": unique(f"keyword:{path}/{entry['name']}"), "kind": "keyword", "name": entry["name"], "path": path,
                  "title": f"{path}/{entry['name']}", "summary": _first_sentence(text), "text": text[:MAX_TEXT],
                  "also_in": item["count"] - 1}
        record.update({key: entry[key] for key in ("aliases", "type", "n_var", "usage", "default", "unit", "lone", "deprecated")
                       if entry[key]})
        if entry["choices"]:
            record["choices"] = [{"name": choice["name"], "description": _cp2k_links(choice["description"])[:400]}
                                 for choice in entry["choices"][:200]]
        if entry["repeats"]:
            record["repeats"] = True
        if entry["removed"]:
            record["removed"] = True
        records.append(record)
    return version, records


def _strip_comment(line):
    return re.split(r"(?:^|(?<=\s))[#!]", line, maxsplit=1)[0].rstrip()


def _symbol(name):
    letters = re.sub(r"[^A-Za-z].*", "", name)
    return _element(letters[:1].upper() + letters[1:2].lower()) or _element(letters[:1].upper())


def parse_cp2k_input(relative, source):
    """One CP2K input: a digest of the settings that matter for planning, its elements and features."""
    variables = {name.upper(): value for name, value in re.findall(r"(?im)^\s*@SET\s+(\w+)\s+(.+?)\s*$", source)}

    def expand(value):
        return re.sub(r"\$\{?(\w+)\}?", lambda m: variables.get(m.group(1).upper(), m.group(0)), value)
    stack, values, present, symbols, comments, lines = [], {}, set(), set(), [], []
    atoms = shown = 0
    for raw in source.splitlines():
        stripped = raw.strip()
        if not stack and not values and stripped.startswith(("#", "!")):
            comments.append(stripped.lstrip("#!").strip())
        line = _strip_comment(raw).strip()
        upper = line.upper()
        if stack and stack[-1] == "COORD" and line and not upper.startswith("&"):
            atoms, shown = atoms + 1, shown + 1
            symbol = _symbol(line.split()[0])
            symbols.update([symbol] if symbol else [])
            if shown == 7:
                lines.append(re.match(r"\s*", raw).group() + "...")
            if shown > 6:
                continue
        elif upper.startswith("&END"):
            stack = stack[:-1]
        elif upper.startswith("&"):
            name, _, argument = line[1:].partition(" ")
            stack.append(name.upper())
            present.add("/".join(stack))
            if argument.strip():
                values["/".join(stack)] = expand(argument.strip())
                if stack[-1] == "KIND":
                    symbols.update([_symbol(argument.strip())] if _symbol(argument.strip()) else [])
            shown = 0
        elif line and not upper.startswith("@"):
            keyword, _, value = line.partition(" ")
            key = "/".join([*stack, keyword.upper()])
            present.add(key)
            values.setdefault(key, expand(value.strip()))
            if stack and stack[-1] == "KIND" and keyword.upper() == "ELEMENT" and _element(value.strip()):
                symbols.add(value.strip())
        lines.append(raw)
    elements = sorted(symbol for symbol in symbols if symbol)

    def find(*suffixes):
        return next((value for key, value in values.items() if key.endswith(suffixes)), None)

    def has(*fragments):
        return any(path.endswith(fragments) or any(f"/{fragment}/" in f"/{path}/" for fragment in fragments) for path in present)
    digest = []
    for label, suffixes in (("RUN_TYPE", ("GLOBAL/RUN_TYPE",)), ("METHOD", ("FORCE_EVAL/METHOD",)), ("QS METHOD", ("QS/METHOD",)),
                            ("ENSEMBLE", ("MD/ENSEMBLE",)), ("TIMESTEP", ("MD/TIMESTEP",)), ("TEMPERATURE", ("MD/TEMPERATURE",)),
                            ("STEPS", ("MD/STEPS",)), ("THERMOSTAT", ("THERMOSTAT/TYPE",)),
                            ("OPTIMIZER", ("GEO_OPT/OPTIMIZER", "CELL_OPT/OPTIMIZER")), ("CELL_OPT TYPE", ("CELL_OPT/TYPE",)),
                            ("BAND_TYPE", ("BAND/BAND_TYPE",)), ("CUTOFF", ("MGRID/CUTOFF",)), ("REL_CUTOFF", ("MGRID/REL_CUTOFF",)),
                            ("EPS_SCF", ("SCF/EPS_SCF",)), ("ADDED_MOS", ("SCF/ADDED_MOS",)), ("SMEAR", ("SMEAR/METHOD",)),
                            ("ELECTRONIC_TEMPERATURE", ("SMEAR/ELECTRONIC_TEMPERATURE",)), ("MIXING", ("MIXING/METHOD",)),
                            ("KPOINTS", ("KPOINTS/SCHEME",)), ("XC_FUNCTIONAL", ("XC/XC_FUNCTIONAL",)),
                            ("VDW", ("PAIR_POTENTIAL/TYPE",)), ("PERIODIC", ("CELL/PERIODIC",)), ("ABC", ("CELL/ABC",))):
        value = find(*suffixes)
        if value:
            digest.append(f"{label} {' '.join(value.split())[:60]}")
    features = {"ot": has("SCF/OT"), "diagonalization": has("SCF/DIAGONALIZATION"), "smearing": has("SCF/SMEAR"),
                "kpoints": has("DFT/KPOINTS"), "dft+u": has("DFT_PLUS_U") or "PLUS_U" in source.upper(),
                "hybrid": has("XC/HF"), "admm": has("AUXILIARY_DENSITY_MATRIX_METHOD"), "vdw": has("VDW_POTENTIAL"),
                "md": has("MOTION/MD"), "geo_opt": has("MOTION/GEO_OPT"), "cell_opt": has("MOTION/CELL_OPT"),
                "neb": has("MOTION/BAND"), "tddfpt": has("PROPERTIES/TDDFPT"), "uks": has("DFT/UKS", "DFT/LSD"),
                "dos": has("PRINT/DOS", "PRINT/PDOS"), "bands": has("BAND_STRUCTURE"), "efield": has("PERIODIC_EFIELD", "DFT/EFIELD"),
                "metadynamics": has("FREE_ENERGY/METADYN"), "vibrational": has("VIBRATIONAL_ANALYSIS"), "stm": has("PRINT/STM"),
                "barostat": has("MD/BAROSTAT"), "real-time": has("REAL_TIME_PROPAGATION")}
    if atoms:
        digest.append(f"{atoms} atoms" + (f" ({', '.join(elements)})" if elements else ""))
    summary = "; ".join(filter(None, [" ".join(" ".join(comments).split())[:300], ", ".join(digest)]))
    return {"id": "example:" + relative, "kind": "example", "name": PurePosixPath(relative).name, "title": relative,
            "summary": summary[:700], "text": "\n".join(lines)[:MAX_TEXT], "elements": elements,
            "tags": sorted(name for name, on in features.items() if on), "source": relative}


def parse_cp2k_markdown(relative, source):
    """A page of CP2K's documentation, split at its second-level headings."""
    text = re.sub(r"```\{(?:youtube|figure|image|raw)\}.*?```", "", source, flags=re.S)
    text = re.sub(r"^(?::::.*|\([^)\s]*\)=)$", "", _cp2k_links(text), flags=re.M)
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.M)
    page = title_match.group(1).strip() if title_match else PurePosixPath(relative).stem
    tags = [part.replace("_", " ").replace("-", " ") for part in PurePosixPath(relative).parts[1:-1]]
    return _markdown_records(relative, text, page, tags)


def _markdown_records(relative, text, page, tags, summary=None):
    """Guide records of one Markdown page, split at its second-level headings outside code blocks."""
    chunks, current, heading = [], [], page
    fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            fence = not fence
        if not fence and re.match(r"^##\s+", line):
            chunks.append((heading, current))
            current, heading = [], line.lstrip("#").strip()
            continue
        if not fence and re.match(r"^#\s+", line):
            continue
        current.append(line)
    chunks.append((heading, current))
    records, slugs = [], set()
    for heading, body in chunks:
        content = re.sub(r"\n{3,}", "\n\n", "\n".join(body)).strip()
        if len(content) < 20:
            continue
        slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-") or "page"
        while slug in slugs:
            slug += "-2"
        slugs.add(slug)
        title = page if heading == page else f"{page}: {heading}"
        for index in range(0, len(content), 6000):
            part = content[index:index + 6000]
            suffix = "" if index == 0 else f"-{index // 6000 + 1}"
            records.append({"id": f"guide:{relative}#{slug}{suffix}", "kind": "guide", "name": heading, "title": title,
                            "summary": summary if summary and not records else _first_sentence(re.sub(r"```.*?```", "", part, flags=re.S)),
                            "text": part, "tags": tags, "source": relative})
    return records


def build_cp2k_documents(*, cache_dir=None, timeout=120):
    """Download CP2K's pinned documentation pages and example inputs and index them."""
    documents = catalog()["cp2k"]["documents"]
    cache = _downloads("cp2k-docs", cache_dir) / documents["tag"]
    items = [{"name": item["path"].replace("/", "__"), "url": documents["base"] + item["path"],
              "sha256": item["sha256"], "size": item["size"]} for item in documents["files"]]
    paths = _fetch_many(items, cache, timeout)
    records = []
    for item, path in zip(documents["files"], paths):
        source = path.read_text(encoding="utf-8", errors="replace")
        if item["path"].endswith(".md"):
            records += parse_cp2k_markdown(item["path"], source)
        else:
            records.append(parse_cp2k_input(item["path"], source))
    version = documents["tag"].lstrip("v")
    payload = {"schema_version": 1, "engine": "cp2k", "part": "documents", "version": version, "built_at": _now(),
               "sources": [{"url": documents["base"], "files": len(items)}], "license": catalog()["cp2k"]["license"],
               "records": records}
    _write(root() / f"cp2k-docs-{documents['tag']}.json", payload)
    return {"engine": "cp2k", "part": "documents", "version": version, "records": len(records),
            "examples": sum(record["kind"] == "example" for record in records)}


def cp2k_input_payload(data):
    """The index of a gzip-compressed cp2k --xml output."""
    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
        xml = stream.read(MAX_XML_BYTES + 1)
    if len(xml) > MAX_XML_BYTES:
        raise ValueError("CP2K's input reference is unexpectedly large.")
    version, records = parse_cp2k_xml(xml)
    if not re.fullmatch(r"\d+(?:\.\d+)+", version):
        raise ValueError("CP2K's input reference does not state its version.")
    return {"schema_version": 1, "engine": "cp2k", "part": "keywords", "version": version, "built_at": _now(),
            "sources": [{"program": "cp2k --xml", "sha256": hashlib.sha256(xml).hexdigest()}],
            "license": catalog()["cp2k"]["license"], "records": records}


def build_cp2k_keywords(config, transport=None):
    """Run cp2k --xml on the login node and index the keyword reference of exactly that build."""
    from .engine_setup import _connection
    program = _program(config, _CP2K_PROGRAMS)
    folder = str(PurePosixPath(config.remote_root) / ".dft-agent" / ("reference-" + uuid4().hex))
    quoted = shlex.quote(folder)
    command = "\n".join(["set -e", *config.setup_commands, "export OMP_NUM_THREADS=1", f"mkdir -p {quoted}", f"cd {quoted}",
                         "rm -f cp2k_input.xml cp2k_input.xml.gz",
                         f"if ! {shlex.quote(program)} --xml > xml.log 2>&1; then tail -n 20 xml.log >&2; exit 3; fi",
                         "gzip -f cp2k_input.xml"])
    with _connection(config, transport) as active, tempfile.TemporaryDirectory(prefix="dft-cp2k-xml-") as directory:
        result = active.run(command, timeout=300)
        if result.returncode:
            raise ValueError("cp2k --xml did not run on the login node: " + (result.stderr or result.stdout).strip()[-600:])
        local = Path(directory) / "cp2k_input.xml.gz"
        active.download(folder + "/cp2k_input.xml.gz", local)
        active.run(f"rm -f {quoted}/cp2k_input.xml.gz {quoted}/xml.log; rmdir {quoted}", timeout=60)
        payload = cp2k_input_payload(local.read_bytes())
    _write(root() / f"cp2k-input-{payload['version']}.json", payload)
    _bind(config, "cp2k", payload["version"])
    return {"engine": "cp2k", "part": "keywords", "version": payload["version"], "records": len(payload["records"])}


# Elk and pw.x print their versions in a banner before they miss their input; abinit --version prints
# only the version.
_VERSION_CHECK = {"elk": ("< /dev/null", r"Elk code version (\d+(?:\.\d+)+)"), "abinit": ("--version", r"(?m)^\s*(\d+\.\d+\.\d+)\s*$"),
                  "qe": ("< /dev/null", r"Program PWSCF v\.?\s*(\d+(?:\.\d+)+)")}


def _program_version(config, engine, transport=None):
    from .engine_setup import _connection
    if engine == "qe":
        programs = [token for token in shlex.split(getattr(config, "qe_command", "") or "")
                    if PurePosixPath(token).name == "pw.x"]
        program = programs[-1] if programs else "pw.x"
        if getattr(config, "qe_bin_dir", ""):
            program = str(PurePosixPath(config.qe_bin_dir) / "pw.x")
    else:
        program = _program(config, (engine,))
    arguments, pattern = _VERSION_CHECK[engine]
    # One thread: an OpenMP build otherwise starts a thread per login-node core just to print its banner.
    command = "\n".join([*config.setup_commands, "export OMP_NUM_THREADS=1", 'folder=$(mktemp -d)', 'cd "$folder"',
                         f"timeout 60 {shlex.quote(program)} {arguments} 2>&1 | head -n 20 || true", "cd /", 'rm -rf "$folder"'])
    with _connection(config, transport) as active:
        output = active.run(command, timeout=120).stdout
    match = re.search(pattern, output)
    if not match:
        raise ValueError(f"{LABELS[engine]} did not report its version on the login node.")
    return match.group(1)


def build_for_environment(config, transport=None, *, cache_dir=None):
    """Everything this environment's program has: the matching Elk, ABINIT or QE manual, or CP2K's documents and keywords."""
    engine = config.engine
    if engine in _BUILDERS:
        version = environment_version(engine, config)
        if version is None:
            version = _program_version(config, engine, transport)
            _bind(config, engine, version)
        built = _BUILDERS[engine](version, cache_dir=cache_dir)
        built["environment_version"] = version
        return [built]
    if engine == "cp2k":
        return [build_cp2k_documents(cache_dir=cache_dir), build_cp2k_keywords(config, transport)]
    raise ValueError("References are available for Elk, ABINIT, CP2K and QE environments.")


# The programs whose manuals come from pinned releases, and how each is built.
_BUILDERS = {"elk": build_elk, "abinit": build_abinit, "qe": build_qe}

# --- search --------------------------------------------------------------------

_STOP = frozenset("a an and are as at be by can for from has have how i in into is it its may of on or our that the "
                  "their them then there these this to use used using was we were what when which will with you your".split())
_WORD = re.compile(r"[a-z0-9]+(?:[_+-][a-z0-9]+)*")
# Chinese requests meet an English manual through these terms.
_TERMS = {
    "能带": "band structure", "态密度": "density of states dos", "投影": "projected pdos", "光学": "optical optics",
    "吸收": "absorption", "介电": "dielectric", "激子": "exciton bootstrap", "激光": "laser pulse", "退磁": "demagnetization",
    "反铁磁": "antiferromagnetic", "铁磁": "ferromagnetic", "磁": "magnetic magnetism", "自旋轨道": "spin-orbit soc",
    "自旋": "spin", "声子": "phonon", "超导": "superconductivity eliashberg", "电声": "electron-phonon coupling",
    "分子动力学": "molecular dynamics md", "动力学": "dynamics", "弛豫": "relaxation geometry optimization",
    "结构优化": "geometry optimization", "晶胞": "cell lattice", "温度": "temperature", "恒温": "thermostat nvt",
    "液": "liquid", "水": "water", "表面": "surface slab", "缺陷": "defect vacancy", "空位": "vacancy",
    "杂化泛函": "hybrid functional hse pbe0", "色散": "dispersion vdw d3", "范德华": "van der waals vdw",
    "费米面": "fermi surface", "有效质量": "effective mass", "电荷密度": "charge density", "功函数": "work function",
    "电场": "electric field", "极化": "polarization", "压电": "piezoelectric", "各向异性": "anisotropy",
    "高次谐波": "high harmonic generation hhg", "谐波": "harmonic", "时间演化": "time evolution", "含时": "time-dependent tddft",
    "收敛": "convergence", "截断": "cutoff", "k点": "k-point kpoints", "布里渊区": "brillouin zone",
    "赝势": "pseudopotential", "基组": "basis set", "带隙": "band gap", "能隙": "band gap", "金属": "metal smearing",
    "绝缘": "insulator", "半导体": "semiconductor", "大体系": "large system orbital transformation ot",
    "超胞": "supercell", "振动": "vibrational", "红外": "infrared", "拉曼": "raman", "过渡态": "transition state neb",
    "反应路径": "reaction path neb", "扩散": "diffusion", "径向分布": "radial distribution rdf", "压力": "pressure stress",
    "应力": "stress", "状态方程": "equation of state", "准粒子": "quasiparticle gw", "光谱": "spectrum", "X射线": "x-ray",
    "磁矩": "magnetic moment", "磁化": "magnetization", "非共线": "noncollinear", "自旋螺旋": "spin spiral",
    "费米能": "fermi energy", "电子温度": "electronic temperature smearing", "混合": "mixing", "平衡": "equilibration",
}


def _normal(word):
    for old, new in (("isation", "ization"), ("ising", "izing"), ("ised", "ized")):
        if word.endswith(old):
            word = word[:-len(old)] + new
            break
    if len(word) > 6 and word.endswith("ise"):
        word = word[:-3] + "ize"
    if len(word) > 4 and word.endswith("s") and not word.endswith(("ss", "us", "is", "ics")):
        word = word[:-1]
    return word


def _tokens(text):
    result = []
    for word in _WORD.findall(str(text).lower()):
        parts = [part for part in re.split(r"[_+-]", word) if part]
        if len(parts) > 1:
            result.append(word)
        for part in parts:
            if part in _STOP:
                continue
            part = _normal(part)
            result.append(part)
            if len(part) >= 6 and not part.isdigit():
                result.append("~" + part[:5])
    return result


# Everyday phrases meet the programs' own names for them.
_SYNONYMS = {
    "molecular dynamics": "md", "aimd": "md molecular dynamics", "water": "h2o", "geometry optimization": "geo_opt",
    "geometry optimisation": "geo_opt", "relaxation": "geo_opt optimization", "relax": "geo_opt optimization",
    "cell optimization": "cell_opt", "variable cell": "cell_opt", "density of states": "dos", "band structure": "band_structure",
    "hybrid": "hf admm", "dispersion": "vdw_potential dftd3", "van der waals": "vdw_potential dftd3", "d3": "dftd3",
    "d4": "dftd4", "nudged elastic band": "neb band", "k-point": "kpoints", "kpoint": "kpoints", "smearing": "smear",
    "dft+u": "plus_u dft_plus_u", "hubbard": "plus_u dft_plus_u", "metal": "smear diagonalization",
    "large system": "orbital transformation ot", "orbital transformation": "ot", "spin polarized": "uks lsd",
    "spin-polarized": "uks lsd", "thermostat": "csvr nose", "excited state": "tddfpt", "laser": "pulse",
}


def _expand(text):
    lowered = str(text).lower()
    extra = [english for chinese, english in _TERMS.items() if chinese in text]
    extra += [names for phrase, names in _SYNONYMS.items() if re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", lowered)]
    return " ".join([text, *extra])


_PRINT = re.compile(r"(?:^|/)(?:PRINT|EACH)(?:/|$)")
_FIELDS = (("name", 4.0), ("aliases", 4.0), ("title", 2.0), ("tags", 2.0), ("summary", 1.5), ("text", 1.0), ("choices", 1.0))


class Reference:
    """Records of one program, merged from its manual, documents, keyword reference and lessons."""

    def __init__(self, engine, payloads, notes=()):
        self.engine, self.notes, self.records, self.parts = engine, list(notes), [], {}
        for payload in payloads:
            part = payload.get("part", "manual")
            self.parts[part] = {"version": payload.get("version"), "built_at": payload.get("built_at"),
                                "records": len(payload.get("records", []))}
            self.records += [{**record, "part": part, "version": payload.get("version")} for record in payload.get("records", [])]
        for lesson in catalog()[engine].get("lessons", []):
            self.records.append({"id": "lesson:" + lesson["id"], "kind": "lesson", "name": lesson["id"], "title": lesson["title"],
                                 "summary": _first_sentence(lesson["text"]), "text": lesson["text"], "tags": lesson.get("tags", []),
                                 "elements": lesson.get("elements", []), "source": lesson.get("evidence", ""), "part": "lessons"})
        self.by_id = {record["id"]: record for record in self.records}
        self.postings, self.lengths = defaultdict(list), []
        for position, record in enumerate(self.records):
            counts = Counter()
            for field, weight in _FIELDS:
                value = record.get(field)
                if isinstance(value, list):
                    value = " ".join(f"{item.get('name', '')} {item.get('description', '')}" if isinstance(item, dict) else str(item)
                                     for item in value)
                for token in _tokens(str(value or "")[:MAX_TEXT]):
                    counts[token] += weight * (0.5 if token.startswith("~") else 1.0)
            for token, count in counts.items():
                self.postings[token].append((position, count))
            self.lengths.append(sum(counts.values()) or 1.0)
        self.average = sum(self.lengths) / max(1, len(self.lengths))

    def search(self, query, *, kinds=None, elements=(), limit=8):
        terms = set(_tokens(_expand(str(query))[:8000]))
        total, scores = len(self.records), defaultdict(float)
        for term in terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            weight = math.log(1 + (total - len(posting) + 0.5) / (len(posting) + 0.5)) * (0.5 if term.startswith("~") else 1.0)
            for position, count in posting:
                scores[position] += weight * count * 2.2 / (count + 1.2 * (0.25 + 0.75 * self.lengths[position] / self.average))
        words = {word.lower() for word in re.findall(r"[A-Za-z0-9_./-]+", str(query))}
        printing = bool(re.search(r"(?i)print|output|write|file", str(query)))
        wanted, ranked = set(elements or ()), []
        for position, score in scores.items():
            record = self.records[position]
            if kinds and record["kind"] not in kinds:
                continue
            names = {str(record.get("name", "")).lower(), *(str(alias).lower() for alias in record.get("aliases", []))}
            # A named input variable or task; examples are matched by their text and elements instead.
            if record["kind"] in {"keyword", "section", "task"} and names & words:
                score += 8.0
            if record.get("path") and record["id"].split(":", 1)[1].lower() in words:
                score += 12.0
            if wanted and record.get("elements"):
                score += 3.0 * len(wanted & set(record["elements"])) / len(set(record["elements"]))
            score += record.get("prior", 0.0)
            if record.get("path"):
                # GLOBAL, FORCE_EVAL and MOTION before the atomic code, tests and rarely used branches;
                # output controls only when the question is about output.
                rank, depth, _ = _path_rank(record["path"])
                score += (3.0, 1.0, -1.0, -4.0)[rank] - 0.5 * max(0, depth - 2)
                if not printing and _PRINT.search(record["path"]):
                    score -= 6.0
            ranked.append((score, record["id"], record))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [(record, score) for score, _, record in ranked[:max(1, min(limit, 50))]]

    def find(self, identifier):
        """A record by id, by bare name or CP2K path, or the best match."""
        text = str(identifier).strip()
        if text in self.by_id:
            return self.by_id[text]
        lowered = text.lower()
        if self.engine == "cp2k":
            lowered = re.sub(r"^cp2k_input/", "", lowered.lstrip("&").replace(".", "/"))
        for kind in ("keyword", "section", "task", "example", "guide", "lesson"):
            for record in self.records:
                if record["kind"] != kind:
                    continue
                if record["id"].split(":", 1)[-1].lower() == lowered:
                    return record
        for record in self.records:
            if record["kind"] in {"keyword", "task"} and lowered in {str(record["name"]).lower(), *map(str.lower, record.get("aliases", []))}:
                return record
        hits = self.search(text, limit=1)
        return hits[0][0] if hits else None

    def source(self, record):
        version = record.get("version")
        label = LABELS[self.engine] + (f" {version}" if version else "")
        kind = record["kind"]
        if kind == "lesson":
            return "DFT Agent cluster check: " + record["title"]
        if self.engine == "elk":
            return {"keyword": f"{label} manual, input block {record['name']}", "task": f"{label} manual, task {record['name']}",
                    "guide": f"{label} manual, {record['name']}", "example": f"{label} example {record['title']}"}.get(kind, label)
        if self.engine == "abinit" and kind == "keyword":
            return f"{label} {'' if record.get('program') == 'abinit' else record.get('program', '') + ' '}input variable {record['name']}"
        if self.engine == "qe" and kind in {"keyword", "section"}:
            return f"{label} {record['program']} input: " + (f"{record['namelist']} {record['name']}" if kind == "keyword" else record["name"])
        if kind in {"keyword", "section"}:
            return f"{label} input reference (cp2k --xml): {record['title']}"
        return f"{label} {'documentation' if kind == 'guide' else 'example'}: {record['title']}"

    def url(self, record):
        if record["kind"] == "lesson":
            return None
        if self.engine == "elk":
            return catalog()["elk"]["manual"]
        if self.engine == "qe":
            entry, source = catalog()["qe"], PurePosixPath(record.get("source", ""))
            if record["kind"] in {"keyword", "section"}:
                return f"{entry['manual']}{source.stem}.html" + (f"#{record['name']}" if record["kind"] == "keyword" else "")
            commit = entry["releases"].get(record.get("version"), {}).get("commit", "develop")
            return f"{entry['repository']}/blob/{commit}/{source}"
        if self.engine == "abinit":
            entry, source = catalog()["abinit"], PurePosixPath(record.get("source", ""))
            if record["kind"] == "keyword":
                return f"{entry['manual']}variables/{record.get('varset', record.get('program', 'basic'))}/#{record['name']}"
            if record["kind"] == "guide":
                page = source.stem.lstrip("_")
                return f"{entry['manual']}{source.parent.name}/{page}/"
            commit = catalog()["abinit"]["releases"].get(record.get("version"), {}).get("commit", "master")
            return f"{entry['repository']}/blob/{commit}/{source}"
        if record["kind"] in {"guide", "example"}:
            return "https://github.com/cp2k/cp2k/blob/" + catalog()["cp2k"]["documents"]["tag"] + "/" + record["source"]
        return catalog()["cp2k"]["manual"]

    def excerpt(self, record, limit=1400):
        lines = [record["title"]]
        if record["kind"] == "keyword":
            facts = [f"{key} {record[key]}" for key in ("type", "default", "unit", "dimensions", "status", "requires") if record.get(key)]
            if facts:
                lines.append("; ".join(facts))
            if record.get("choices"):
                lines.append("Choices: " + ", ".join(choice["name"] for choice in record["choices"][:30]))
        if record["kind"] == "example" and record.get("summary"):
            lines.append(record["summary"])
        if record["kind"] == "example" and self.engine in {"abinit", "qe"}:
            labels = _ABINIT_OUTSIDE if self.engine == "abinit" else _QE_OUTSIDE
            outside = [label for tag, label in labels.items() if tag in record.get("tags", [])]
            if outside:
                lines.append(f"Uses what DFT Agent's {LABELS[self.engine]} does not offer: " + ", ".join(outside))
        if record["kind"] == "example" and self.engine == "elk" and record.get("tasks"):
            from .elk import NATIVE_TASKS
            described = []
            for number in record["tasks"][:8]:
                task = next((item for item in self.records if item["kind"] == "task" and number in item.get("numbers", [])), None)
                described.append(f"{number} ({task['summary'][:90]})" if task else str(number))
            lines.append("Tasks: " + "; ".join(described))
            outside = [str(number) for number in record["tasks"] if number not in NATIVE_TASKS]
            if outside:
                lines.append("Not available in DFT Agent's Elk tasks: " + ", ".join(outside))
        body = record.get("text", "")
        if record["kind"] == "example" and self.engine == "elk":
            body = "\n".join(line for line in body.splitlines() if not line.lstrip().startswith("!"))
        if record["kind"] == "example" and self.engine == "abinit":
            # The leading comments are the summary above; comments beside values stay.
            body = re.sub(r"\A(?:[ \t]*(?:[#!].*)?\n)+", "", body)
        lines.append(body)
        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        if len(text) > limit:
            text = text[:limit].rsplit("\n", 1)[0] + "\n..."
        return text


_LOADED = {}


def _payload(path):
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key not in _LOADED:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != 1 or not isinstance(value.get("records"), list):
            raise ValueError(f"{path.name} is not a reference index.")
        if len(_LOADED) > 8:
            _LOADED.clear()
        _LOADED[key] = value
    return _LOADED[key]


_REFERENCES = {}


def _build_once(name, build):
    """Build an index unless the last attempt failed within the hour; return a note on failure."""
    marker = root() / f".failed-{name}"
    try:
        if marker.is_file() and time.time() - marker.stat().st_mtime < RETRY_SECONDS:
            return f"The {name} reference could not be downloaded recently; it is retried after an hour."
    except OSError:
        pass
    try:
        build()
        marker.unlink(missing_ok=True)
        return None
    except Exception as exc:  # noqa: BLE001 - a missing reference never stops planning
        try:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(f"{type(exc).__name__}: {exc}"[:500], encoding="utf-8")
        except OSError:
            pass
        return f"The {name} reference download failed. Retry with Get manuals and examples."


def load(engine, config=None, *, build=True, cache_dir=None, timeout=120):
    """The reference for the version an environment runs; downloads pinned texts on first use."""
    if engine not in ENGINES:
        raise ValueError("References are available for Elk, ABINIT, CP2K and QE.")
    wanted, notes, paths = environment_version(engine, config), [], []
    build = build and not os.environ.get("DFT_AGENT_NO_REFERENCE_DOWNLOAD")
    if engine in _BUILDERS:
        version, label = release(engine, wanted), LABELS[engine]
        if wanted and version != wanted:
            notes.append(f"No manual for {label} {wanted} is pinned; this is {label} {version}'s.")
        path = root() / f"{engine}-{version}.json"
        if not path.is_file() and build:
            builder = _BUILDERS[engine]
            note = _build_once(f"{engine}-{version}", lambda: builder(version, cache_dir=cache_dir, timeout=timeout))
            notes += [note] if note else []
        paths += [path] if path.is_file() else []
    else:
        tag = catalog()["cp2k"]["documents"]["tag"]
        documents = root() / f"cp2k-docs-{tag}.json"
        if not documents.is_file() and build:
            note = _build_once(f"cp2k-docs-{tag}", lambda: build_cp2k_documents(cache_dir=cache_dir, timeout=timeout))
            notes += [note] if note else []
        paths += [documents] if documents.is_file() else []
        built = sorted(root().glob("cp2k-input-*.json"), key=lambda item: _version_key(item.stem))
        exact = root() / f"cp2k-input-{wanted}.json" if wanted else None
        if exact is not None and exact.is_file():
            paths.append(exact)
        elif built:
            paths.append(built[-1])
            notes.append(f"Keyword reference of CP2K {built[-1].stem.split('-')[-1]}; this environment runs "
                         f"{'CP2K ' + wanted if wanted else 'a version not checked yet'}.")
        else:
            notes.append("CP2K's keyword reference is not built yet; build it for the environment (runs cp2k --xml on the login node).")
        if wanted and tag.lstrip("v") != wanted:
            notes.append(f"The CP2K guides and examples are from {tag.lstrip('v')}; this environment runs {wanted}.")
    payloads = []
    for path in paths:
        try:
            payloads.append(_payload(path))
        except (OSError, ValueError) as exc:
            notes.append(f"{path.name} could not be read: {exc}"[:200])
    key = (engine, tuple((str(path), path.stat().st_mtime_ns) for path in paths if path.is_file()), tuple(notes))
    if key not in _REFERENCES:
        if len(_REFERENCES) > 8:
            _REFERENCES.clear()
        _REFERENCES[key] = Reference(engine, payloads, notes)
    return _REFERENCES[key]


def available(engine, config=None):
    """Which parts of an environment's reference exist, from the files alone."""
    wanted = environment_version(engine, config)
    if engine in _BUILDERS:
        version = release(engine, wanted)
        return {"version": wanted or version, "manual": (root() / f"{engine}-{version}.json").is_file()}
    tag = catalog()["cp2k"]["documents"]["tag"]
    return {"version": wanted, "documents": (root() / f"cp2k-docs-{tag}.json").is_file(),
            "keywords": bool(wanted) and (root() / f"cp2k-input-{wanted}.json").is_file()}


def status(engine, config=None):
    """What is built for an environment, without downloading anything."""
    reference = load(engine, config, build=False)
    return {"engine": engine, "environment_version": environment_version(engine, config), "parts": reference.parts,
            "notes": reference.notes}


def search(engine, query, *, config=None, kinds=None, elements=(), limit=8):
    reference = load(engine, config)
    kinds = [kind for kind in (kinds or []) if kind in KINDS] or None
    hits = reference.search(query, kinds=kinds, elements=elements, limit=min(int(limit), MAX_RESULTS))
    return {"engine": engine, "parts": reference.parts, "notes": reference.notes,
            "results": [{"id": record["id"], "kind": record["kind"], "title": record["title"],
                         "summary": record.get("summary", "")[:300], "source": reference.source(record), "score": round(score, 2)}
                        for record, score in hits]}


def read(engine, identifier, *, config=None):
    reference = load(engine, config)
    record = reference.find(identifier)
    if record is None:
        raise ValueError(f"No {LABELS[engine]} reference entry matches {identifier!r}.")
    value = {key: record[key] for key in ("id", "kind", "name", "title", "program", "namelist", "path", "type", "default", "unit",
                                          "status", "dimensionality", "see", "usage", "n_var", "dimensions", "varset", "requires",
                                          "excludes", "characteristics", "aliases",
                                          "fields", "choices", "keywords", "subsections", "parameters", "tasks", "blocks",
                                          "variables", "chain", "elements", "tags", "deprecated", "removed") if record.get(key)}
    value.update(source=reference.source(record), url=reference.url(record), text=record.get("text", "")[:MAX_READ])
    if record["kind"] != "lesson":
        value["license"] = catalog()[engine]["license"]["name"]
    if record["kind"] == "example":
        value["note"] = "An official demonstration, not a converged production setting."
    return value


_ELK_WORDS = frozenset("tasks atoms scale scissor molecule species tetra".split())


def planner_context(engine, goal, *, elements=(), config=None, history=None, limit=4):
    """A few excerpts for the planner: verified lessons, manual sections and examples, and native names the user wrote."""
    try:
        reference = load(engine, config, timeout=QUICK_TIMEOUT)
    except Exception:  # noqa: BLE001 - planning goes on without references
        return []
    text = " ".join([str(goal), *[str(item.get("content", "")) for item in history or []
                                  if isinstance(item, dict) and item.get("role") == "user"]])[:6000]
    # Native names the user wrote: CP2K's upper-case keywords, or Elk blocks, ABINIT variables and QE
    # variables and cards that are not ordinary words.
    if engine == "cp2k":
        native = set(re.findall(r"\b[A-Z][A-Z0-9_]{1,}\b", text))
    else:
        words, pattern = {"elk": (_ELK_WORDS, r"\b[a-z][a-z0-9_]{3,}\b"), "abinit": (_ABINIT_WORDS, r"\b[a-z][a-z0-9_]{2,}\b"),
                          "qe": (_QE_WORDS, r"\b[A-Za-z][A-Za-z0-9_]{2,}\b")}[engine]
        native = {word.lower() for word in re.findall(pattern, text)} - words
    quota = {"lesson": 2, "guide": 2, "example": 2, "task": 1, "keyword": 2, "section": 1}
    hits = reference.search(text, elements=elements, limit=40)
    if engine in {"elk", "abinit", "qe"}:
        # Examples of what DFT Agent runs come before those it cannot run.
        from .elk import NATIVE_TASKS

        def runnable(record):
            if engine == "elk":
                return set(record.get("tasks", [])) <= NATIVE_TASKS
            return not set(record.get("tags", [])) & set(_ABINIT_OUTSIDE if engine == "abinit" else _QE_OUTSIDE)
        hits = sorted(((record, score * (1.5 if record["kind"] == "example" and runnable(record) else 1.0))
                       for record, score in hits), key=lambda item: -item[1])
    if engine == "qe":
        # A variable name the user wrote means pw.x's when pw.x has it, not dos.x's or matdyn.x's.
        best = Counter()
        for record, score in hits:
            if record["kind"] == "keyword":
                best[record["name"].lower()] = max(best[record["name"].lower()], score)
        hits = sorted(((record, best[record["name"].lower()] + 0.01 if record["kind"] == "keyword" and record.get("program") == "pw.x"
                        else score) for record, score in hits), key=lambda item: -item[1])
    # Official texts are compared with each other: a keyword the user named, or a short lesson that
    # matches every word, would otherwise crowd them out.
    best = max((score for record, score in hits if record["kind"] in {"guide", "example", "task"}), default=0.0)
    top = hits[0][1] if hits else 0.0
    picked, counts, names = [], Counter(), set()
    for record, score in hits:
        kind = record["kind"]
        path = record.get("path") or ""
        if counts[kind] >= quota[kind]:
            continue
        if kind in {"keyword", "section"}:
            # One entry per name: QE's programs and CP2K's sections repeat many of them. QE's card
            # columns carry plain words (Density, label) and are left to the card's own entry.
            if str(record["name"]).lower() in names or engine == "qe" and kind == "keyword" and not str(record.get("namelist")).startswith("&"):
                continue
            # ABINIT has over a thousand variables, many for development: only those the user named.
            named = record["name"] in native if engine == "cp2k" else str(record["name"]).lower() in native
            if not named and (engine == "abinit" or score < 0.9 * top) or path and (_path_rank(path)[0] > 1 or _PRINT.search(path)):
                continue
        # Example inputs carry little prose, so they rank lower than manual text that matches as well;
        # they must share an element with the structure.
        elif score < max(4.0, (0.4 if kind == "example" else 0.6) * best) or (
                kind == "example" and elements and record.get("elements") and not set(elements) & set(record["elements"])):
            continue
        counts[kind] += 1
        names.add(str(record["name"]).lower())
        picked.append(record)
        if len(picked) >= limit:
            break
    return [{"source": reference.source(record), "text": reference.excerpt(record),
             **({"notes": reference.notes} if reference.notes else {})} for record in picked]
