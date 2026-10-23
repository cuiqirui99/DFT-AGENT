"""Connecting an API model: the free key check, regions, model choice and provider replies."""

import pytest

from vasp_slurm_agent import agent, model_connect
from vasp_slurm_agent.model_connect import REGIONS, ConnectFailure


class Reply:
    def __init__(self, status, payload=None, text=""):
        self.status_code, self.payload, self.text = status, payload, text

    def json(self):
        if self.payload is None:
            raise ValueError("not JSON")
        return self.payload


def server(answers, calls=None):
    """A fake HTTP GET answering by URL prefix."""
    def get(url, headers, timeout, local):
        if calls is not None:
            calls.append((url, headers, local))
        for prefix, answer in answers.items():
            if url.startswith(prefix):
                if isinstance(answer, Exception):
                    raise answer
                return answer
        raise AssertionError(url)
    return get


def test_pasted_keys_lose_spaces_and_quotes_and_other_providers_are_recognised():
    assert model_connect.clean_key('  "sk-abc 123\n" ') == "sk-abc123"
    assert model_connect.key_owner("responses", "sk-ant-api03-x") == "anthropic"
    assert model_connect.key_owner("anthropic", "sk-ant-api03-x") is None
    assert model_connect.key_owner("deepseek", "xai-123") == "grok"
    assert model_connect.key_owner("responses", "0123456789abcdef0123456789abcdef.ABCDEFGHIJKLMNOP") == "glm"
    assert model_connect.key_owner("deepseek", "sk-123") is None


def test_regions_put_mainland_china_first_for_a_chinese_interface():
    assert [name for name, _ in model_connect.regions("qwen", "zh")][0] == "Beijing"
    assert [name for name, _ in model_connect.regions("qwen", "en")][0] == "Singapore"
    assert [name for name, _ in model_connect.regions("glm", "zh")] == ["BigModel", "Z.AI"]
    assert model_connect.regions("responses") == ((None, "https://api.openai.com/v1"),)


@pytest.mark.parametrize("reply,expected", [
    (Reply(200, {"data": [{"id": "gpt-6.1-sol"}, {"id": "text-embedding-3"}, {"bad": 1}]}), ("ok", ["gpt-6.1-sol", "text-embedding-3"])),
    (Reply(401, text='{"error":{"code":"invalid_api_key"}}'), ("rejected", [])),
    (Reply(403), ("denied", [])),
    (Reply(429), ("rate", [])),
    (Reply(402), ("credit", [])),
    # xAI answers a wrong key with 400.
    (Reply(400, text='{"code":"invalid-argument","error":"Incorrect API key provided."}'), ("rejected", [])),
    (Reply(404), ("unsupported", [])),
    (Reply(200, None), ("unsupported", [])),
    (Reply(500), ("error", [])),
])
def test_model_list_replies(reply, expected):
    assert model_connect.list_models("grok", "xai-key", "https://api.x.ai/v1", get=server({"https://api.x.ai": reply})) == expected


def test_model_list_headers_and_network_failures():
    import httpx
    calls = []
    model_connect.list_models("anthropic", "sk-ant-key", "https://api.anthropic.com",
                              get=server({"https://api.anthropic.com": Reply(200, {"data": []})}, calls))
    url, headers, local = calls[0]
    assert url == "https://api.anthropic.com/v1/models?limit=100" and headers == {"x-api-key": "sk-ant-key", "anthropic-version": "2023-06-01"}
    model_connect.list_models("deepseek", "sk-key", "https://api.deepseek.com/", get=server({"https://api.deepseek.com": Reply(200, {"data": []})}, calls))
    assert calls[1][:2] == ("https://api.deepseek.com/models", {"Authorization": "Bearer sk-key"}) and not calls[1][2]
    model_connect.list_models("local", "", "http://127.0.0.1:11434/v1", get=server({"http://127.0.0.1": Reply(200, {"data": []})}, calls))
    assert calls[2][1] == {} and calls[2][2]  # no key, and no proxy for this computer
    failed = server({"https://api.deepseek.com": httpx.ConnectError("refused")})
    assert model_connect.list_models("deepseek", "sk-key", "https://api.deepseek.com", get=failed) == ("unreachable", [])


def test_malformed_catalog_ids_cannot_be_selected_for_a_retry():
    invalid = ["a" * 201, "bad model", "", "bad\x00model"]
    get = server({"https://api.openai.com": Reply(200, {"data": [{"id": name} for name in [*invalid, "gpt-5"]]})})
    assert model_connect.list_models("responses", "key", "https://api.openai.com/v1", get=get) == ("ok", ["gpt-5"])
    assert model_connect.choose_model("chat_completions", [*invalid, "valid"]) == "valid"


@pytest.mark.parametrize("provider,models,requested,expected", [
    ("deepseek", ["deepseek-flash", "deepseek-v4-pro"], "deepseek-flash", "deepseek-flash"),
    ("deepseek", ["deepseek-v4-pro", "deepseek-chat"], "deepseek-flash", "deepseek-v4-pro"),
    ("deepseek", [], "deepseek-flash", "deepseek-flash"),  # no list: keep what was asked for
    ("responses", ["gpt-5-mini", "gpt-5", "gpt-5-2025-08-07", "text-embedding-3-large", "gpt-image-1"], "gpt-6.1-sol", "gpt-5"),
    ("responses", ["gpt-6.2-sol", "gpt-6.1-sol", "gpt-5"], "gpt-7", "gpt-6.2-sol"),
    ("anthropic", ["claude-haiku-5-5", "claude-sonnet-5-5", "claude-opus-5-5"], "claude-sonnet-4-6", "claude-sonnet-5-5"),
    ("qwen", ["qwen-max", "qwen-plus", "qwen-vl-plus", "text-embedding-v4"], "qwen-turbo", "qwen-plus"),
    ("glm", ["glm-5.3", "glm-5.3-flash", "glm-4.6v"], "glm-6", "glm-5.3"),
    ("grok", ["grok-4.7", "grok-4.20-0309-reasoning", "grok-3"], "grok-5", "grok-4.7"),
    ("chat_completions", ["kimi-k3", "kimi-embedding"], "", "kimi-k3"),
    ("chat_completions", ["Llama-3.3-70B-Instruct", "text-embedding"], "missing", "Llama-3.3-70B-Instruct"),
    ("local", ["text-embedding"], "missing", ""),
])
def test_a_model_the_account_has(provider, models, requested, expected):
    assert model_connect.choose_model(provider, models, requested) == expected


def test_a_key_is_tried_in_each_region_until_one_accepts_it():
    beijing, singapore = REGIONS["qwen"][0][1], REGIONS["qwen"][1][1]
    calls = []
    get = server({beijing: Reply(401), singapore: Reply(200, {"data": [{"id": "qwen-plus"}]}), "https://": Reply(401)}, calls)
    found = model_connect.find_endpoint("qwen", "sk-key", language="zh", get=get)
    assert found == {"base_url": singapore, "region": "Singapore", "models": ["qwen-plus"], "model": "qwen-plus"}
    assert [url for url, _, _ in calls] == [beijing + "/models", singapore + "/models"]


def test_bigmodel_keys_keep_their_address_and_preset_addresses_are_not_stored():
    bigmodel, zai = REGIONS["glm"][0][1], REGIONS["glm"][1][1]
    found = model_connect.find_endpoint("glm", "k.k", language="en",
                                        get=server({zai: Reply(401), bigmodel: Reply(200, {"data": [{"id": "glm-5.3"}]})}))
    assert found["base_url"] == bigmodel and found["region"] == "BigModel"
    found = model_connect.find_endpoint("responses", "sk-key", get=server({"https://api.openai.com": Reply(200, {"data": [{"id": "gpt-6.1-sol"}]})}))
    assert found["base_url"] is None and found["model"] == "gpt-6.1-sol"


def test_failures_carry_what_the_person_should_do():
    import httpx
    with pytest.raises(ConnectFailure) as caught:
        model_connect.find_endpoint("responses", "sk-ant-key", get=server({"https://": Reply(401)}))
    assert caught.value.code == "key" and caught.value.details == {"owner": "anthropic"}
    with pytest.raises(ConnectFailure) as caught:
        model_connect.find_endpoint("deepseek", "sk-key", get=server({"https://": httpx.ConnectTimeout("slow")}))
    assert caught.value.code == "network" and caught.value.details == {"host": "api.deepseek.com"}
    with pytest.raises(ConnectFailure) as caught:
        model_connect.find_endpoint("chat_completions", "sk-key", base_url="https://llm.example.org/v1", get=server({"https://": Reply(404)}))
    assert caught.value.code == "model"


def test_local_servers_are_found_with_their_chat_models():
    import httpx
    get = server({"http://127.0.0.1:11434": Reply(200, {"data": [{"id": "qwen3:8b"}, {"id": "nomic-embed-text"}]}),
                  "http://127.0.0.1": httpx.ConnectError("refused")})
    assert model_connect.local_servers(get=get) == [{"name": "Ollama", "base_url": "http://127.0.0.1:11434/v1", "models": ["qwen3:8b"]}]


@pytest.mark.parametrize("address", ["https://user:secret@example.org/v1", "https://example.org/v1?key=secret",
    "https://example.org/#secret", "https://example.org:99999/v1", "https://example.org:bad/v1",
    "https://example.org:0/v1", "https://ex ample.org/v1", "https://example.org/\n", "file:///tmp/models"])
def test_invalid_addresses_are_rejected_before_sending_a_key(address):
    calls = []
    with pytest.raises(ConnectFailure) as caught:
        model_connect.find_endpoint("chat_completions", "private-key", base_url=address, model="test", get=lambda *args: calls.append(args))
    assert caught.value.code == "address"
    assert "secret" not in str(caught.value) and "private-key" not in str(caught.value)
    assert not calls


def test_local_address_is_checked_before_discovery():
    calls = []
    with pytest.raises(ConnectFailure, match="valid API address"):
        model_connect.find_endpoint("local", "", base_url="https://remote.example.org/v1", get=lambda *args: calls.append(args))
    assert not calls


def test_restricted_catalog_still_allows_a_real_model_probe():
    found = model_connect.find_endpoint("responses", "restricted-key", model="gpt-5",
                                       get=server({"https://api.openai.com": Reply(403)}))
    assert found["model"] == "gpt-5" and found["models"] == []


@pytest.mark.parametrize("status,code", [(429, "rate"), (402, "credit")])
def test_catalog_errors_retain_actionable_codes(status, code):
    with pytest.raises(ConnectFailure) as caught:
        model_connect.find_endpoint("responses", "key", get=server({"https://api.openai.com": Reply(status)}))
    assert caught.value.code == code


class Failure(Exception):
    def __init__(self, status, error):
        super().__init__("provider error")
        self.status_code, self.body = status, {"error": error}


@pytest.mark.parametrize("status,error,code", [
    (401, {"message": "Incorrect API key provided"}, "key"),
    (400, {"code": "invalid-argument", "message": "Incorrect API key provided. You can obtain an API key from https://console.x.ai."}, "key"),
    (400, {"message": "Model Not Exist", "type": "invalid_request_error"}, "model"),
    (404, {"message": "The model `gpt-9` does not exist"}, "model"),
    (402, {"message": "Insufficient Balance"}, "credit"),
    (400, {"code": "Arrearage", "message": "Access denied, please make sure your account is in good standing."}, "credit"),
    (429, {"type": "insufficient_quota"}, "credit"),
    (429, {"message": "Rate limit reached"}, "rate"),
    (403, {"message": "forbidden"}, "denied"),
    (400, {"message": "response_format json_schema is not supported"}, "format"),
    (503, {"message": "overloaded"}, "server"),
])
def test_provider_replies_get_a_code(status, error, code):
    assert agent._provider_error(Failure(status, error), "deepseek")[0] == code
