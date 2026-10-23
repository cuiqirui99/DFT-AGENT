"""Code that runs on the cluster uses the cluster's own python3, which can be as old as 3.6.

Arrhenius (EL9) has 3.9: restart.py's Path.write_text(newline=...), from 3.10, stopped every VASP hybrid
warm start there. These checks cover the uploaded modules and the scripts embedded as strings.
"""
import ast
import importlib
from pathlib import Path

import pytest

PACKAGE = Path(__file__).parents[1] / "src" / "vasp_slurm_agent"
FILES = ("restart.py", "dispatch.py", "scheduler.py")
EMBEDDED = (("engine_workflow", "_RUNTIME"), ("qe_workflow", "_RUNTIME"), ("qe_workflow", "_UPF_VALIDATOR"),
            ("script_workflow", "_EXECUTE"), ("qe_setup", "_BUILD_PY"), ("engine_setup", "_STAGE_PY"),
            ("engine_setup", "_INSTALL_PY"), ("abinit", "LIBRARY_VALIDATOR"), ("elk", "LIBRARY_VALIDATOR"),
            ("cp2k", "LIBRARY_VALIDATOR"), ("transport", "_HASH_CODE"), ("transport", "_CHECKED_ARCHIVE_CODE"))
# Keyword arguments newer than Python 3.6: subprocess.run(capture_output/text) 3.7,
# Path.write_text/read_text(newline) 3.10/3.13, zip(strict) 3.10.
NEWER = {"capture_output": None, "text": "run", "newline": {"write_text", "read_text"}, "strict": "zip"}


def _sources():
    for name in FILES:
        yield name, (PACKAGE / name).read_text(encoding="utf-8")
    for module, attribute in EMBEDDED:
        yield f"{module}.{attribute}", getattr(importlib.import_module(f"vasp_slurm_agent.{module}"), attribute)


@pytest.mark.parametrize("name,source", list(_sources()), ids=lambda value: value if isinstance(value, str) and len(value) < 60 else "")
def test_cluster_code_avoids_newer_python(name, source):
    tree = ast.parse(source, feature_version=(3, 7))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        called = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
        for keyword in node.keywords:
            scope = NEWER.get(keyword.arg, False)
            if scope is None or scope == called or (isinstance(scope, set) and called in scope):
                pytest.fail(f"{name}:{node.lineno}: {called}({keyword.arg}=...) needs a newer python3 than some clusters have")
