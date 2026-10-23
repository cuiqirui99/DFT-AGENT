"""Find an API endpoint and model before testing a real request."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlsplit

from .providers import PROVIDERS

# Where each provider's accounts, keys and API credit are managed. Qwen and GLM run separate
# platforms for mainland China and elsewhere; a key works only on its own.
GUIDES = {
    "responses": {"site": "platform.openai.com", "keys": "https://platform.openai.com/api-keys",
                  "billing": "https://platform.openai.com/settings/organization/billing/overview"},
    "anthropic": {"site": "platform.claude.com", "keys": "https://platform.claude.com/settings/keys",
                  "billing": "https://platform.claude.com/settings/billing"},
    "deepseek": {"site": "platform.deepseek.com", "keys": "https://platform.deepseek.com/api_keys",
                 "billing": "https://platform.deepseek.com/top_up"},
    "qwen": {"site": "bailian.console.aliyun.com", "keys": "https://bailian.console.aliyun.com/?apiKey=1",
             "billing": "https://bailian.console.aliyun.com/",
             "abroad": ("modelstudio.console.alibabacloud.com", "https://modelstudio.console.alibabacloud.com/ap-southeast-1/settings/api-key")},
    "glm": {"site": "bigmodel.cn", "keys": "https://bigmodel.cn/usercenter/proj-mgmt/apikeys",
            "billing": "https://bigmodel.cn/finance/overview",
            "abroad": ("z.ai", "https://z.ai/manage-apikey/apikey-list")},
    "grok": {"site": "console.x.ai", "keys": "https://console.x.ai/", "billing": "https://console.x.ai/"},
}
REGIONS = {
    "qwen": (("Beijing", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
             ("Singapore", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"),
             ("Virginia", "https://dashscope-us.aliyuncs.com/compatible-mode/v1"),
             ("Hong Kong", "https://cn-hongkong.dashscope.aliyuncs.com/compatible-mode/v1")),
    "glm": (("BigModel", "https://open.bigmodel.cn/api/paas/v4"), ("Z.AI", "https://api.z.ai/api/paas/v4")),
}
# Models tried, best first, when the account lacks the one asked for.
PREFERRED = {
    "responses": (r"gpt-6", r"gpt-5", r"gpt-4\.1", r"gpt-4o"),
    "anthropic": (r"claude-sonnet", r"claude-opus", r"claude-haiku"),
    "deepseek": (r"deepseek-flash$", r"deepseek-v4-pro$", r"deepseek-chat$"),
    "qwen": (r"qwen-plus$", r"qwen-max$", r"qwen3-max", r"qwen-plus", r"qwen3", r"qwen-turbo$"),
    "glm": (r"glm-5\.\d+$", r"glm-5", r"glm-4\.\d+$", r"glm-4"),
    "grok": (r"grok-4\.\d+$", r"grok-4", r"grok-3"),
}
_NOT_CHAT = re.compile(r"embed|rerank|tts|audio|speech|whisper|transcri|realtime|image|dall-e|vision|moderation|search|ocr|asr|"
                       r"video|babbage|davinci|codex|computer-use|omni|-vl", re.I)
_SNAPSHOT = re.compile(r"-\d{4}-?\d{2}-?\d{2}$|-\d{4}$|-(?:preview|beta|exp)")
_SMALL = re.compile(r"mini|nano|lite|tiny|small", re.I)
# Signs of a key from another provider; most keys start with "sk-", so only distinctive ones count.
_KEY_SHAPES = (("anthropic", re.compile(r"^sk-ant-")), ("grok", re.compile(r"^xai-")),
               ("glm", re.compile(r"^[0-9a-f]{32}\.[0-9A-Za-z]{16}$")))
_LOCAL_SERVERS = (("Ollama", "http://127.0.0.1:11434/v1"), ("LM Studio", "http://127.0.0.1:1234/v1"),
                  ("llama.cpp", "http://127.0.0.1:8080/v1"), ("vLLM", "http://127.0.0.1:8000/v1"))


class ConnectFailure(ValueError):
    """A connection that cannot work yet, with a code the interface explains."""

    def __init__(self, code, message, **details):
        super().__init__(message)
        self.code, self.details = code, details


def clean_key(text):
    """The key as pasted, without the spaces, line breaks or quotes copying tends to add."""
    return re.sub(r"\s+", "", str(text or "")).strip("'\"")


def key_owner(provider, key):
    """The provider a key looks like it belongs to, when that is not the selected one."""
    for owner, shape in _KEY_SHAPES:
        if shape.search(key) and owner != provider:
            return owner
    return None


def regions(provider, language="en"):
    """(name, base URL) pairs to try for a key; mainland China first for a Chinese interface."""
    found = REGIONS.get(provider)
    if not found:
        return ((None, PROVIDERS[provider]["base_url"] or "https://api.openai.com/v1"),)
    china = [item for item in found if item[0] in {"Beijing", "BigModel"}]
    rest = [item for item in found if item not in china]
    return tuple(china + rest) if language == "zh" else tuple(rest + china)


def _get(url, headers, timeout, local):
    import httpx

    with httpx.Client(timeout=timeout, trust_env=not local, follow_redirects=False) as client:
        return client.get(url, headers=headers)


def list_models(provider, key, base_url, *, timeout=20.0, get=None):
    """Read the model catalog without generating tokens."""
    _check_address(provider, base_url)
    base_url = base_url.rstrip("/")
    if provider == "anthropic":
        url, headers = base_url.rstrip("/") + "/v1/models?limit=100", {"x-api-key": key, "anthropic-version": "2023-06-01"}
    else:
        url, headers = base_url + "/models", ({"Authorization": f"Bearer {key}"} if key else {})
    try:
        response = (get or _get)(url, headers, timeout, _loopback(base_url))
    except Exception as error:  # noqa: BLE001 - every transport failure means the same to the person
        names = {kind.__name__ for kind in type(error).__mro__}
        return ("unreachable", []) if names & {"ConnectError", "ConnectTimeout", "TimeoutException", "NetworkError",
                                               "ProxyError", "OSError", "ReadTimeout"} else ("error", [])
    text = response.text[:2000] if hasattr(response, "text") else ""
    if response.status_code == 401 or (response.status_code == 400 and re.search(r"(?:incorrect|invalid|wrong)[ _-]+api[ _-]?key", text, re.I)):
        return "rejected", []
    if response.status_code == 403:
        # A restricted key may allow inference but not model discovery.
        return "denied", []
    if response.status_code == 402:
        return "credit", []
    if response.status_code == 429:
        return "rate", []
    if response.status_code in {404, 405}:
        return "unsupported", []
    if response.status_code != 200:
        return "error", []
    try:
        data = response.json().get("data", [])
        return "ok", [item["id"] for item in data if isinstance(item, dict) and _valid_model(item.get("id"))]
    except (ValueError, AttributeError, TypeError):
        return "unsupported", []


def _loopback(base_url):
    host = urlsplit(base_url).hostname or ""
    try:
        return host == "localhost" or ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _check_address(provider, address):
    try:
        if not isinstance(address, str) or any(c.isspace() or ord(c) < 32 for c in address):
            raise ValueError
        parsed = urlsplit(address)
        if (parsed.scheme not in {"http", "https"} or not parsed.hostname
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError
        if parsed.port is not None and parsed.port < 1:
            raise ValueError
        if provider == "local" and not _loopback(address):
            raise ValueError
    except ValueError:
        raise ConnectFailure("address", "Enter a valid API address without credentials or query parameters.") from None


def _version(name):
    return tuple(int(part) for part in re.findall(r"\d+", name))


def _valid_model(name):
    return isinstance(name, str) and 0 < len(name) <= 200 and not any(c.isspace() or ord(c) < 32 for c in name)


def choose_model(provider, models, requested=""):
    """The model to use: the one asked for when the account lists it (or lists nothing), else the
    best general chat model it has."""
    if requested and (not models or requested in models):
        return requested
    chat = [name for name in models if _valid_model(name) and not _NOT_CHAT.search(name)]
    for pattern in PREFERRED.get(provider, ()):
        matching = [name for name in chat if re.match(pattern, name)]
        if matching:
            return sorted(matching, key=lambda name: (bool(_SMALL.search(name)), bool(_SNAPSHOT.search(name)),
                                                      tuple(-part for part in _version(name)), len(name)))[0]
    return chat[0] if chat else ""


def find_endpoint(provider, key, *, base_url="", model="", language="en", get=None):
    """Where this key works and which model to use there, or a ConnectFailure.

    {"base_url", "region", "models", "model"}; base_url is None when the provider's preset address works."""
    candidates = ((None, base_url),) if base_url else regions(provider, language)
    if provider == "chat_completions" and base_url and not base_url.rstrip("/").endswith("/v1"):
        # Providers often give their site address; the API usually lives under /v1.
        candidates = ((None, base_url), (None, base_url.rstrip("/") + "/v1"))
    outcomes = []
    for region, address in candidates:
        status, models = list_models(provider, key, address, get=get)
        outcomes.append((status, region, address, models))
        if status == "ok":
            break
    # An address that lists models wins; one without a model list may still answer requests.
    usable = ([item for item in outcomes if item[0] == "ok"]
              or [item for item in outcomes if item[0] == "unsupported"]
              or [item for item in outcomes if item[0] == "denied"])
    if usable:
        _, region, address, models = usable[0]
        chosen = choose_model(provider, models, model or PROVIDERS[provider]["model_example"])
        if not chosen:
            raise ConnectFailure("model", "Enter the model name this provider gave you.")
        preset = PROVIDERS[provider]["base_url"] or (regions(provider)[0][1] if provider == "responses" else None)
        # Qwen, compatible and local connections always record their address.
        keep = address != preset or provider in {"qwen", "chat_completions", "local"}
        return {"base_url": address if keep else None, "region": region, "models": models, "model": chosen}
    outcomes = [item[0] for item in outcomes]
    if "rate" in outcomes:
        raise ConnectFailure("rate", "The service is limiting requests. Try again later.")
    if "credit" in outcomes:
        raise ConnectFailure("credit", "Check the account's API credit.")
    if "rejected" in outcomes:
        owner = key_owner(provider, key)
        raise ConnectFailure("key", "The provider did not accept this key.", owner=owner)
    if all(status == "unreachable" for status in outcomes):
        raise ConnectFailure("network", "The provider could not be reached.",
                             host=urlsplit(candidates[0][1]).hostname or "")
    raise ConnectFailure("server", "The provider did not answer the check.")


def local_servers(*, get=None):
    """Model servers running on this computer, with their models."""
    found = []
    for name, address in _LOCAL_SERVERS:
        status, models = list_models("local", "", address, timeout=2.0, get=get)
        if status == "ok":
            found.append({"name": name, "base_url": address, "models": [item for item in models if not _NOT_CHAT.search(item)]})
    return found
