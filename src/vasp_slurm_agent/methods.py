"""Explicit VASP method settings. Magnetic moments use the uploaded site order."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np


METHOD_DEFAULTS = {
    "spin": "none", "magmom": None, "soc": False, "saxis": [0.0, 0.0, 1.0],
    "functional": "PBE", "hubbard_u": {},
}
HYBRID_BAND_NELMIN = 10


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number.")
    return float(value)


def normalize_method(parameters: dict[str, Any], species: list[str]) -> dict[str, Any]:
    method = {key: parameters.get(key, value) for key, value in METHOD_DEFAULTS.items()}
    spin, soc = method["spin"], method["soc"]
    if spin not in {"none", "collinear", "noncollinear"}:
        raise ValueError("spin must be none, collinear or noncollinear.")
    if type(soc) is not bool:
        raise ValueError("soc must be a boolean.")
    if method["functional"] not in {"PBE", "HSE06", "PBE0"}:
        raise ValueError("functional must be PBE, HSE06 or PBE0.")
    if soc and spin == "collinear":
        raise ValueError("SOC requires spin=noncollinear with vector moments, or spin=none.")
    axis = method["saxis"]
    if not isinstance(axis, (list, tuple)) or len(axis) != 3:
        raise ValueError("saxis must contain three numbers.")
    axis = [_number(value, "saxis") for value in axis]
    length = float(np.linalg.norm(axis))
    if not math.isfinite(length) or length <= 1e-12:
        raise ValueError("saxis must be nonzero.")
    axis = [value / length for value in axis]
    if not (soc or spin == "noncollinear") and not np.allclose(axis, [0, 0, 1]):
        raise ValueError("saxis applies only to SOC or noncollinear calculations.")
    method["saxis"] = axis
    moments = method["magmom"]
    if spin == "none":
        if moments is not None:
            raise ValueError("Set a spin mode before supplying magmom.")
        method["magmom"] = [[0.0, 0.0, 0.0] for _ in species] if soc else None
    else:
        if not isinstance(moments, (list, tuple)) or len(moments) != len(species):
            raise ValueError("magmom must provide one moment per uploaded site.")
        if spin == "collinear":
            method["magmom"] = [_number(value, "magmom") for value in moments]
        else:
            if any(not isinstance(value, (list, tuple)) or len(value) != 3 for value in moments):
                raise ValueError("Noncollinear magmom needs three components per site in the SAXIS spinor basis.")
            method["magmom"] = [[_number(component, "magmom") for component in value] for value in moments]
    hubbard = method["hubbard_u"]
    if not isinstance(hubbard, dict) or any(element not in species for element in hubbard):
        raise ValueError("hubbard_u must map elements present in the structure to l, u and j.")
    normalized = {}
    for element, values in hubbard.items():
        if not isinstance(values, dict) or set(values) != {"l", "u", "j"}:
            raise ValueError(f"hubbard_u[{element}] requires explicit l, u and j.")
        orbital = values["l"]
        if type(orbital) is not int or orbital not in {0, 1, 2, 3}:
            raise ValueError("Hubbard l must be 0, 1, 2 or 3.")
        u, j = _number(values["u"], "U"), _number(values["j"], "J")
        if not 0 <= j <= u:
            raise ValueError("Dudarev settings require 0 <= J <= U (eV).")
        normalized[element] = {"l": orbital, "u": u, "j": j}
    method["hubbard_u"] = normalized
    return method


def method_incar(method: dict[str, Any], elements: list[str], task: str) -> dict[str, Any]:
    ncl = method["soc"] or method["spin"] == "noncollinear"
    tags: dict[str, Any] = {"ISPIN": 2 if method["spin"] == "collinear" else 1,
                            "ISYM": 0 if task == "bands" else 3 if method["functional"] != "PBE" else 2}
    if method["magmom"] is not None:
        tags["MAGMOM"] = np.asarray(method["magmom"]).reshape(-1).tolist()
        tags["LORBIT"] = 11
    if ncl:
        tags.update(LNONCOLLINEAR=True, LSORBIT=method["soc"], SAXIS=method["saxis"], GGA_COMPAT=False, ISYM=-1)
    hubbard = method["hubbard_u"]
    if hubbard:
        tags.update(
            LDAU=True, LDAUTYPE=2,
            LDAUL=[hubbard.get(element, {}).get("l", -1) for element in elements],
            LDAUU=[hubbard.get(element, {}).get("u", 0.0) for element in elements],
            LDAUJ=[hubbard.get(element, {}).get("j", 0.0) for element in elements],
            LMAXMIX=max(2, 2 * max(value["l"] for value in hubbard.values())),
        )
    if method["functional"] != "PBE":
        tags.update(LHFCALC=True, AEXX=0.25, HFSCREEN=0.2 if method["functional"] == "HSE06" else 0.0,
                    ALGO="Damped", TIME=0.4, ICHARG=2, ISTART=0)
        if method["functional"] == "PBE0":
            # A truncated Coulomb kernel speeds the k-point convergence of unscreened exchange. With HSE06's
            # screening VASP 5.4.4 applied the truncated bare kernel instead: silicon came out with PBE0's gaps
            # (4.0 eV at Γ, 0.7 eV above QE's HSE06 on the same mesh).
            tags["HFRCUT"] = -1
        if task == "bands":
            # Davidson optimizes empty path states that Damped can leave unconverged.
            tags.update(ALGO="Normal", IMIX=1, AMIX=0.2, LFOCKACE=True,
                        NELMIN=HYBRID_BAND_NELMIN, ISYM=-1)
            tags.pop("TIME")
    return tags


def method_fingerprint(method: dict[str, Any], species: list[str], potcar_labels: list[str], *, comparison: bool = False) -> str:
    selected = {key: value for key, value in method.items() if not comparison or key not in {"spin", "magmom"}}
    payload = {"method": selected, "species": species, "potcar_labels": potcar_labels}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def validate_method_output(incar: dict[str, Any], metadata: dict[str, Any], *, vasp_version: str = "") -> None:
    """Compare the XML's actual INCAR with the frozen method recipe."""
    expected = metadata.get("method_incar_expected")
    if expected is None:
        if metadata.get("method") is not None:
            raise ValueError("Prepared method checks are missing from metadata.")
        # Legacy runs had no method metadata and supported only nonmagnetic PBE.
        if int(incar.get("ISPIN", 1)) != 1 or any(incar.get(key, False) for key in ("LSORBIT", "LNONCOLLINEAR", "LDAU", "LHFCALC")):
            raise ValueError("Method metadata is required for this calculation.")
        return
    for tag, planned in expected.items():
        actual = incar.get(tag, False if type(planned) is bool and not planned else None)
        if actual is None:
            # VASP 5 has no adaptively compressed exchange: it ignores LFOCKACE and its Davidson iterations
            # apply the full exchange operator. Silicon's HSE06 path on 5.4.4 converged in 12 steps.
            if tag == "LFOCKACE" and planned is True and vasp_version.startswith("5."):
                continue
            raise ValueError(f"Method mismatch: {tag} is missing from output INCAR.")
        # VASP 5 writes this scalar as a one-element XML vector.
        if (tag == "LDAUTYPE" and vasp_version.startswith("5.") and type(planned) is int
                and isinstance(actual, (list, tuple)) and len(actual) == 1 and type(actual[0]) is int):
            actual = actual[0]
        if isinstance(planned, str):
            matches = str(actual).lower() == planned.lower()
        elif type(planned) is bool:
            matches = type(actual) is bool and actual == planned
        else:
            left, right = np.asarray(actual), np.asarray(planned)
            if tag == "MAGMOM":
                left, right = left.reshape(-1), right.reshape(-1)
            matches = left.shape == right.shape and np.allclose(left, right, rtol=1e-7, atol=1e-8)
        if not matches:
            raise ValueError(f"Method mismatch: output {tag} differs from the prepared inputs.")
