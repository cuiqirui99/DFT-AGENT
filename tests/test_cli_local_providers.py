"""Provider contracts: isolated CLI reasoning and private local endpoints."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vasp_slurm_agent import agent
from vasp_slurm_agent.agent import AgentError, ModelSettings
from vasp_slurm_agent.model_usage import normalize_usage


SCHEMA = {"type": "object", "properties": {"answer": {"type": "string"}},
          "required": ["answer"], "additionalProperties": False}


def request(settings):
    return agent._request_structured('{"question":"Inspect my result"}', SCHEMA, settings,
                                     instructions="Return the requested JSON. Do not use tools.")


def test_claude_cli_preserves_login_but_disables_tools_and_customizations(monkeypatch, tmp_path):
    monkeypatch.setattr(agent.shutil, "which", lambda value: "/bin/claude")
    for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "DFT_AGENT_SSH_PASSWORD"):
        monkeypatch.setenv(name, "never-forward-this")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(tmp_path / "auth"))
    called = []
    def run(command, **kwargs):
        called.append(command)
        assert command[0] == "/bin/claude" and "-p" in command
        assert command[command.index("--tools") + 1] == ""
        assert command[command.index("--setting-sources") + 1] == ""
        assert json.loads(command[command.index("--mcp-config") + 1]) == {"mcpServers": {}}
        assert json.loads(command[command.index("--settings") + 1]) == {"disableAllHooks": True}
        assert json.loads(command[command.index("--json-schema") + 1]) == SCHEMA
        for flag in ("--safe-mode", "--restricted", "--no-session-persistence", "--strict-mcp-config", "--disable-slash-commands"):
            assert flag in command
        assert "--bare" not in command and "--dangerously-skip-permissions" not in command
        assert "--model" not in command and "Inspect my result" not in " ".join(command)
        assert "Inspect my result" in kwargs["input"]
        assert Path(kwargs["cwd"]).is_dir() and str(tmp_path) not in kwargs["cwd"]
        assert kwargs["env"]["CLAUDE_CONFIG_DIR"] == str(tmp_path / "auth")
        assert "never-forward-this" not in kwargs["env"].values()
        return SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({
            "type": "result", "subtype": "success", "is_error": False,
            "structured_output": {"answer": "Checked"},
            "usage": {"input_tokens": 100, "output_tokens": 20,
                      "cache_read_input_tokens": 30, "cache_creation_input_tokens": 10},
        }))
    monkeypatch.setattr(agent.subprocess, "run", run)
    result = request(ModelSettings("claude_cli"))
    assert json.loads(result) == {"answer": "Checked"}
    assert result.usage["input_tokens"] == 140 and result.usage["total_tokens"] == 160
    assert result.usage["model"] == "Claude Code CLI default"
    assert len(called) == 1


@pytest.mark.parametrize("result,match", [
    ({"subtype": "success", "is_error": False}, "no structured result"),
    ({"subtype": "success", "structured_output": {"answer": 42}}, "invalid structured"),
    ({"subtype": "error_max_structured_output_retries", "is_error": True}, "could not respond"),
    ({"subtype": "error_during_execution", "is_error": True, "errors": ["not logged in: private-marker"]}, "Sign in"),
])
def test_claude_cli_rejects_incomplete_or_failed_results(monkeypatch, result, match):
    monkeypatch.setattr(agent.shutil, "which", lambda _: "/bin/claude")
    monkeypatch.setattr(agent.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=0, stdout=json.dumps(result), stderr=""))
    with pytest.raises(AgentError, match=match) as caught:
        request(ModelSettings("claude_cli", "test-model"))
    assert "private-marker" not in str(caught.value)


@pytest.mark.parametrize("provider", ["codex", "claude_cli"])
@pytest.mark.parametrize("detail,match", [
    ("exceeded retry limit, last status: 429 Too Many Requests private-marker", "rate or usage limit"),
    ("401 unauthorized private-marker", "Sign in"),
    ("model not found private-marker", "model was not found"),
    ("503 service unavailable private-marker", "temporarily unavailable"),
    ("unknown option --safe-mode private-marker", "Update"),
])
def test_cli_error_events_are_classified_without_echo(provider, detail, match, monkeypatch):
    monkeypatch.setattr(agent.shutil, "which", lambda _: "/bin/" + provider)
    event = {"type": "turn.failed", "error": {"message": detail}}
    monkeypatch.setattr(agent.subprocess, "run", lambda *args, **kwargs: SimpleNamespace(
        returncode=1, stdout=json.dumps(event), stderr=""))
    with pytest.raises(AgentError, match=match) as caught:
        request(ModelSettings(provider, "test-model"))
    assert "private-marker" not in str(caught.value)


def test_claude_cli_timeout_and_absent_installation_are_actionable(monkeypatch):
    monkeypatch.setattr(agent.shutil, "which", lambda _: None)
    with pytest.raises(AgentError, match="not found"):
        request(ModelSettings("claude_cli"))
    monkeypatch.setattr(agent.shutil, "which", lambda _: "/bin/claude")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("private-marker", 120)
    monkeypatch.setattr(agent.subprocess, "run", timeout)
    with pytest.raises(AgentError, match="timed out") as caught:
        request(ModelSettings("claude_cli"))
    assert "private-marker" not in str(caught.value)


def test_local_model_uses_real_sdk_schema_without_cloud_credentials(monkeypatch):
    import openai
    from openai import _base_client
    http = getattr(_base_client, "httpx2", None) or _base_client.httpx
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-send-cloud-key")
    monkeypatch.setenv("DFT_AGENT_API_KEY", "do-not-send-other-key")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: do-not-send-header\nX-Private: do-not-send-custom")
    monkeypatch.setenv("OPENAI_ORG_ID", "do-not-send-organization")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "do-not-send-project")
    original_client = openai.DefaultHttpxClient
    calls = []
    def handler(message):
        calls.append(message)
        body = json.loads(message.content)
        assert body["response_format"]["json_schema"]["schema"] == SCHEMA
        assert body["model"] == "local-model" and body["max_tokens"] >= 8192
        assert message.headers["authorization"] == "Bearer local-no-key"
        assert "do-not-send" not in str(message.headers) + message.content.decode()
        return http.Response(200, json={"id": "local", "object": "chat.completion", "created": 1,
            "model": "local-model", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"answer":"Checked"}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}, request=message)
    def client(**kwargs):
        assert kwargs == {"trust_env": False, "follow_redirects": False}
        return original_client(transport=http.MockTransport(handler), **kwargs)
    monkeypatch.setattr(openai, "DefaultHttpxClient", client)
    result = request(ModelSettings("local", "local-model", "http://127.0.0.1:1234/v1"))
    assert json.loads(result)["answer"] == "Checked" and result.usage["total_tokens"] == 15
    assert len(calls) == 1 and str(calls[0].url) == "http://127.0.0.1:1234/v1/chat/completions"


@pytest.mark.parametrize("endpoint", ["https://example.com/v1", "http://192.168.1.5/v1", "http://localhost.evil.test/v1"])
def test_local_provider_cannot_silently_forward_to_remote_endpoint(endpoint):
    with pytest.raises(AgentError, match="loopback"):
        request(ModelSettings("local", "test", endpoint))


def test_connection_resolves_fresh_key_each_request_and_revocation_blocks_cached_key(monkeypatch):
    keys = ["first-test-key", "second-test-key"]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "forbidden-fallback")
    resolved = []
    class Registry:
        def __init__(self, home):
            assert home == "/private/test-registry"
        def model_settings(self, identity, resolve_secret=False):
            assert identity == "connection-id" and resolve_secret
            if not keys:
                raise ValueError("revoked private-marker")
            return ModelSettings("anthropic", "test-model", api_key=keys.pop(0), allow_env=False)
    monkeypatch.setitem(sys.modules, "vasp_slurm_agent.connections", SimpleNamespace(ConnectionRegistry=Registry))
    def call(payload, schema, settings, **kwargs):
        resolved.append(agent._key(settings))
        assert settings.allow_env is False
        return '{"answer":"checked"}'
    monkeypatch.setattr(agent, "_anthropic_request", call)
    settings = ModelSettings("anthropic", "test-model", api_key="stale-cached-key",
                             connection_id="connection-id", connection_home="/private/test-registry", allow_env=False)
    assert agent._settings(settings).connection_id == "connection-id"
    request(settings)
    request(settings)
    assert resolved == ["first-test-key", "second-test-key"]
    with pytest.raises(AgentError, match="revoked") as caught:
        request(settings)
    assert "private-marker" not in str(caught.value)
    assert settings.api_key == "stale-cached-key" and len(resolved) == 2


def test_disabled_environment_fallback_preserved_by_settings(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-use")
    monkeypatch.setenv("DFT_AGENT_BASE_URL", "https://do-not-use.invalid/v1")
    settings = agent._settings(ModelSettings("responses", "explicit-model", allow_env=False))
    assert settings.allow_env is False and settings.base_url is None and agent._key(settings) is None
    with pytest.raises(AgentError, match="API key"):
        request(settings)


def test_saved_openai_key_ignores_sdk_environment_endpoint_and_headers(monkeypatch):
    from test_providers import mock_sdk

    monkeypatch.setenv("OPENAI_BASE_URL", "https://do-not-use.invalid/v1")
    monkeypatch.setenv("OPENAI_CUSTOM_HEADERS", "Authorization: wrong-key\nX-Private: private-marker")
    monkeypatch.setenv("OPENAI_ORG_ID", "wrong-org")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "wrong-project")
    seen = mock_sdk(monkeypatch, "responses", content='{"answer":"Checked"}')
    request(ModelSettings("responses", "test-model", api_key="saved-test-key", allow_env=False))
    assert len(seen) == 1 and str(seen[0].url) == "https://api.openai.com/v1/responses"
    assert seen[0].headers["authorization"] == "Bearer saved-test-key"
    assert "x-private" not in seen[0].headers
    assert "openai-organization" not in seen[0].headers and "openai-project" not in seen[0].headers
