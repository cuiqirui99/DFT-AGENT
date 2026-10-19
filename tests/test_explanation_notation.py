"""Equivalent numerical notation must retain the same evidence requirements."""

import json

import pytest

from vasp_slurm_agent import agent, explanation


@pytest.fixture
def evidence(monkeypatch):
    context = {"facts": {
        "stage_1.residual": {"value": 8.66e-7, "unit": "", "label": "Residual"},
        "stage_1.tolerance": {"value": 1e-6, "unit": "", "label": "Tolerance"},
    }, "context_sha256": "verified", "limits": [], "status": "succeeded"}
    monkeypatch.setattr(explanation, "load_run_context", lambda *args: context)
    return context


def reply(monkeypatch, residual, tolerance):
    text = f"Residual {residual} is below tolerance {tolerance}."
    response = {"answer": text, "evidence": ["stage_1.residual", "stage_1.tolerance"], "limits": [], "next_steps": []}
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: json.dumps(response))
    return text


@pytest.mark.parametrize("residual,tolerance", [
    ("8.66×10⁻⁷", "10⁻⁶"),
    ("8.66 x 10^-7", "10^-6"),
    (r"8.66\times10^{-7}", r"10^{-6}"),
    ("8.66·10⁻⁷", "1×10⁻⁶"),
    ("8.66e-7", "1e-6"),
])
def test_verified_scientific_notation_keeps_original_rendering(evidence, monkeypatch, residual, tolerance):
    expected = reply(monkeypatch, residual, tolerance)
    result = explanation.explain_run("unused", agent.ModelSettings("codex"))
    assert result["answer"] == expected
    assert result["evidence"] == ["stage_1.residual", "stage_1.tolerance"]


@pytest.mark.parametrize("residual", ["8.66×10⁻⁵", "9.66×10⁻⁷", "8.66 x 10^-5", r"8.66\times10^{-5}"])
def test_changed_mantissa_or_exponent_cannot_become_verified(evidence, monkeypatch, residual):
    reply(monkeypatch, residual, "10⁻⁶")
    with pytest.raises(agent.AgentError, match="could not be verified"):
        explanation.explain_run("unused", agent.ModelSettings("codex"))
