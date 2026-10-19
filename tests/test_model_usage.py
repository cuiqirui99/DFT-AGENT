"""Usage comes from provider fields, never character-to-token estimates."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from vasp_slurm_agent import agent, explanation, recovery, structure_agent
from vasp_slurm_agent.agent import AgentError, ModelSettings
from vasp_slurm_agent.model_usage import ModelResponse, codex_usage, normalize_usage, response_provenance, response_usage, usage_record
from test_agent import response as calculation_response, structure
from test_explanation import accepted_run, model_reply
from test_recovery import failed_run, SETTINGS


SCHEMA = agent._object({"answer": {"type": "string"}})


@pytest.mark.parametrize("provider,reported", [
    ("responses", {"input_tokens": 100, "output_tokens": 25,
                   "input_tokens_details": {"cached_tokens": 80}, "output_tokens_details": {"reasoning_tokens": 10}}),
    ("chat_completions", SimpleNamespace(prompt_tokens=100, completion_tokens=25,
        prompt_tokens_details=SimpleNamespace(cached_tokens=80), completion_tokens_details=SimpleNamespace(reasoning_tokens=10))),
    ("codex", {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 25, "reasoning_output_tokens": 10}),
])
def test_normalized_usage_counts_subsets_once(provider, reported):
    result = normalize_usage(reported, provider)
    assert result == {"input_tokens": 100, "output_tokens": 25, "cached_input_tokens": 80,
                      "reasoning_tokens": 10, "total_tokens": 125, "token_counts_source": "provider"}


def test_missing_counts_stay_unknown_and_zero_is_preserved():
    missing = normalize_usage(None, "responses")
    assert all(missing[key] is None for key in ("input_tokens", "output_tokens", "total_tokens", "cached_input_tokens", "reasoning_tokens"))
    assert missing["token_counts_source"] == "unavailable"
    invalid = normalize_usage({"input_tokens": True, "output_tokens": -2}, "responses")
    assert invalid == missing
    zero = normalize_usage({"input_tokens": 0, "output_tokens": 0}, "responses")
    assert zero["total_tokens"] == 0 and zero["token_counts_source"] == "provider"


def test_codex_reads_only_final_turn_usage():
    events = '\n'.join(["not JSON", json.dumps({"type": "item.completed", "usage": {"input_tokens": 999}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 25}}),
        json.dumps({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 25}})])
    assert normalize_usage(codex_usage(events), "codex")["total_tokens"] == 125
    assert codex_usage("{}") is None


def sdk(monkeypatch, provider, *, incomplete=False, reported=True):
    calls = []
    class Client:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=self.create)
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            calls.append(kwargs)
            if provider == "responses":
                return SimpleNamespace(output_text='{"answer":"ok"}', status="incomplete" if incomplete else "completed",
                    usage={"input_tokens": 100, "output_tokens": 25, "input_tokens_details": {"cached_tokens": 80},
                           "output_tokens_details": {"reasoning_tokens": 10}} if reported else None)
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="length" if incomplete else "stop",
                message=SimpleNamespace(content='{"answer":"ok"}', refusal=None))],
                usage={"prompt_tokens": 100, "completion_tokens": 25, "prompt_tokens_details": {"cached_tokens": 80},
                       "completion_tokens_details": {"reasoning_tokens": 10}} if reported else None)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    return calls


@pytest.mark.parametrize("provider", ["responses", "chat_completions"])
def test_sdk_attaches_reported_usage_and_compacts_json(monkeypatch, provider):
    calls = sdk(monkeypatch, provider)
    settings = ModelSettings(provider, "test-model", api_key="private-test-key")
    payload = json.dumps({"goal": "keep  exact spacing", "structure": {"number_of_sites": 2}}, indent=2)
    result = agent._request_structured(payload, SCHEMA, settings, instructions="Return JSON.")
    assert isinstance(result, str) and json.loads(result) == {"answer": "ok"}
    assert isinstance(result, ModelResponse)
    usage = result.usage
    assert usage["input_tokens"] == 100 and usage["total_tokens"] == 125
    assert usage["cached_input_tokens"] == 80 and usage["reasoning_tokens"] == 10
    assert usage["latency_seconds"] >= 0
    compact = calls[0]["input"] if provider == "responses" else calls[0]["messages"][1]["content"]
    assert json.loads(compact) == json.loads(payload) and len(compact) < len(payload)
    assert usage["input_characters"] == len(compact)
    assert usage["schema_characters"] == len(json.dumps(SCHEMA, separators=(",", ":")))
    assert calls[0]["max_output_tokens" if provider == "responses" else "max_completion_tokens"] == 8192
    assert "private-test-key" not in json.dumps(usage)
    again = agent._request_structured(payload, SCHEMA, settings, instructions="Return JSON.")
    assert again.usage["call_id"] != usage["call_id"] and result.usage == usage


@pytest.mark.parametrize("provider", ["responses", "chat_completions"])
def test_length_limited_json_is_not_accepted_and_usage_survives(monkeypatch, provider):
    sdk(monkeypatch, provider, incomplete=True)
    with pytest.raises(AgentError, match="incomplete") as caught:
        agent._request_structured('{"goal":"x"}', SCHEMA, ModelSettings(provider, "test", api_key="private"), instructions="JSON")
    assert caught.value.model_usage["total_tokens"] == 125


def test_cap_scales_for_sites_and_accepts_explicit_budget(monkeypatch):
    monkeypatch.delenv("DFT_AGENT_MAX_OUTPUT_TOKENS", raising=False)
    assert agent._output_limit('{"structure":{"number_of_sites":4096}}') > 8192
    monkeypatch.setenv("DFT_AGENT_MAX_OUTPUT_TOKENS", "4096")
    assert agent._output_limit('{"structure":{"number_of_sites":4096}}') == 4096
    monkeypatch.setenv("DFT_AGENT_MAX_OUTPUT_TOKENS", "-1")
    with pytest.raises(AgentError, match="DFT_AGENT_MAX_OUTPUT_TOKENS"):
        agent._output_limit('{}')


def test_context_budget_rejects_before_provider_without_truncation(monkeypatch):
    calls = sdk(monkeypatch, "responses")
    with pytest.raises(AgentError, match="context is too large"):
        agent._request_structured(json.dumps({"goal": "汉" * 33000}), SCHEMA,
            ModelSettings("responses", "test", api_key="private"), instructions="JSON")
    assert calls == []


def test_codex_json_flag_and_usage_are_compatible_with_output_file(monkeypatch):
    monkeypatch.setattr(agent.shutil, "which", lambda _: "/bin/codex")
    def run(command, **kwargs):
        assert "--json" in command and "--ephemeral" in command and "-o" in command
        Path(command[command.index("-o") + 1]).write_text('{"answer":"ok"}')
        return SimpleNamespace(returncode=0, stdout=json.dumps({"type":"turn.completed", "usage":{
            "input_tokens":100, "cached_input_tokens":80, "output_tokens":25, "reasoning_output_tokens":10}}))
    monkeypatch.setattr(agent.subprocess, "run", run)
    result = agent._request_structured('{"goal":"x"}', SCHEMA, ModelSettings("codex", "test"), instructions="JSON")
    assert result.usage["total_tokens"] == 125 and result.usage["max_output_tokens"] is None


def test_plain_string_mocks_remain_compatible_without_estimated_tokens():
    value = response_usage('{"answer":"ok"}', ModelSettings("codex", "test"))
    assert value["token_counts_source"] == "unavailable" and value["input_tokens"] is None
    assert value["output_characters"] == 15


def test_response_provenance_uses_resolved_connection_and_keeps_legacy_fallback():
    settings = ModelSettings("codex", "old-model", connection_id="saved-connection")
    response = ModelResponse('{}', usage_record("claude_cli", "current-model"))
    assert response_provenance(response, settings) == {"provider": "claude_cli", "model": "current-model"}
    assert response_provenance('{}', ModelSettings("claude_cli")) == {
        "provider": "claude_cli", "model": "Claude Code CLI default"}


def test_saved_connection_model_change_is_recorded_in_actual_plan(monkeypatch, tmp_path, structure):
    from vasp_slurm_agent.connections import ConnectionRegistry

    registry = ConnectionRegistry(tmp_path / "workspace")
    connection = registry.create("model", "Test", {"provider": "responses", "model": "old-model"})
    settings = registry.model_settings(connection["id"])
    registry.update(connection["id"], {"model": "current-model"})
    monkeypatch.setattr(ConnectionRegistry, "_secret", lambda *args: "test-only-key")
    called = []
    class Client:
        def __init__(self, **kwargs):
            self.responses = SimpleNamespace(create=self.create)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            called.append(kwargs["model"])
            return SimpleNamespace(output_text=json.dumps(calculation_response()), status="completed", usage=None)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    plan = agent.draft_plan("Relax.", structure, settings)
    assert settings.model == "old-model" and called == ["current-model"]
    assert plan["provenance"] == {"provider": "responses", "model": "current-model"}
    assert plan["model_usage"]["model"] == "current-model"
    assert json.loads(plan["dialogue"][-1]["content"])["provenance"] == plan["provenance"]


def recorded(text):
    return ModelResponse(json.dumps(text), usage_record("codex", "test", reported={"input_tokens":100,"output_tokens":25}))


def test_calculation_and_structure_results_keep_usage_out_of_dialogue(monkeypatch, structure):
    monkeypatch.setattr(agent, "_request_plan", lambda *args: recorded(calculation_response()))
    plan = agent.draft_plan("Relax.", structure, ModelSettings("codex", "test"))
    assert plan["model_usage"]["total_tokens"] == 125
    assert "model_usage" not in json.loads(plan["dialogue"][-1]["content"])
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: recorded({
        "status":"ready", "summary":"Convert.", "operations":[], "output_format":"poscar", "questions":[], "notes":[]}))
    plan = structure_agent.draft_structure("Convert.", structure, ModelSettings("codex", "test"))
    assert plan["model_usage"]["total_tokens"] == 125
    assert "model_usage" not in json.loads(plan["dialogue"][-1]["content"])


def test_explanation_result_keeps_usage(monkeypatch, accepted_run):
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: recorded(model_reply()))
    answer = explanation.explain_run(accepted_run[0], ModelSettings("codex", "test"))
    assert answer["model_usage"]["total_tokens"] == 125


def test_repair_result_keeps_usage(monkeypatch, failed_run):
    root = failed_run[0]()
    monkeypatch.setattr(agent, "_request_structured", lambda payload, *args, **kwargs: recorded({
        "action":json.loads(payload)["allowed_actions"][0], "diagnosis":"Allow more electronic iterations.", "questions":[]}))
    proposal = recovery.draft_repair(root, SETTINGS)
    assert proposal["model_usage"]["total_tokens"] == 125 and proposal["status"] == "ready"


def test_long_calculation_history_reaches_model_without_losing_first_constraint(monkeypatch, structure):
    history = [{"role": "user", "content": "Never enable SOC."}] + [
        {"role": "user", "content": f"Keep constraint {index}."} for index in range(40)]
    def request(payload, *args):
        assert json.loads(payload)["history"] == history
        return recorded(calculation_response())
    monkeypatch.setattr(agent, "_request_plan", request)
    result = agent.draft_plan("Relax.", structure, ModelSettings("codex", "test"), history)
    assert result["dialogue"][:-1] == history


def test_explanation_preserves_saved_and_current_long_dialogue(monkeypatch, accepted_run):
    root = accepted_run[0]
    proposal_file = root / "proposal.json"
    proposal = json.loads(proposal_file.read_text())
    history = [{"role": "user", "content": "Keep the first constraint."}] + [
        {"role": "user", "content": f"Keep constraint {index}."} for index in range(40)]
    proposal["dialogue"] = history
    proposal_file.write_text(json.dumps(proposal))
    def request(payload, *args, **kwargs):
        data = json.loads(payload)
        assert data["context"]["dialogue"] == history and data["history"] == history
        return recorded(model_reply())
    monkeypatch.setattr(agent, "_request_structured", request)
    assert explanation.explain_run(root, ModelSettings("codex", "test"), history=history)["model_usage"]["total_tokens"] == 125


def test_oversized_explanation_turn_is_rejected_instead_of_cut(monkeypatch, accepted_run):
    monkeypatch.setattr(agent, "_request_structured", lambda *args, **kwargs: pytest.fail("No request allowed"))
    with pytest.raises(AgentError, match="conversation"):
        explanation.explain_run(accepted_run[0], ModelSettings("codex", "test"),
                               history=[{"role":"user", "content":"x" * 20001}])


def test_invalid_model_reply_retains_its_usage(monkeypatch, structure):
    monkeypatch.setattr(agent, "_request_plan", lambda *args: recorded({"bad": "schema"}))
    with pytest.raises(AgentError) as caught:
        agent.draft_plan("Relax.", structure, ModelSettings("codex", "test"))
    assert caught.value.model_usage["total_tokens"] == 125
