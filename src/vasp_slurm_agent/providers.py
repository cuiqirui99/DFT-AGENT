"""Model endpoints and request formats."""

import os


PROVIDERS = {
    "responses": {"label": "OpenAI", "protocol": "responses", "base_url": None,
                  "key_env": "OPENAI_API_KEY", "model_example": "gpt-6.1-sol"},
    "anthropic": {"label": "Claude", "protocol": "anthropic", "base_url": "https://api.anthropic.com",
                  "key_env": "ANTHROPIC_API_KEY", "model_example": "claude-sonnet-4-6"},
    "qwen": {"label": "Qwen", "protocol": "chat_completions", "base_url": None,
             "key_env": "DASHSCOPE_API_KEY", "model_example": "qwen-plus", "json_mode": True,
             "token_parameter": "max_tokens", "extra_body": {"enable_thinking": False}},
    "grok": {"label": "Grok", "protocol": "chat_completions", "base_url": "https://api.x.ai/v1",
             "key_env": "XAI_API_KEY", "model_example": "grok-4.7"},
    "glm": {"label": "GLM", "protocol": "chat_completions", "base_url": "https://api.z.ai/api/paas/v4",
            "key_env": "ZAI_API_KEY", "model_example": "glm-5.3", "json_mode": True,
            "token_parameter": "max_tokens"},
    "deepseek": {"label": "DeepSeek", "protocol": "chat_completions", "base_url": "https://api.deepseek.com",
                 "key_env": "DEEPSEEK_API_KEY", "model_example": "deepseek-flash", "json_mode": True,
                 "token_parameter": "max_tokens", "extra_body": {"thinking": {"type": "disabled"}}},
    "chat_completions": {"label": "Compatible API", "protocol": "chat_completions", "base_url": None,
                         "key_env": "OPENAI_API_KEY", "model_example": ""},
    "codex": {"label": "Codex CLI", "protocol": "codex", "base_url": None,
              "key_env": "", "model_example": ""},
    "claude_cli": {"label": "Claude Code CLI", "protocol": "claude_cli", "base_url": None,
                   "key_env": "", "model_example": ""},
    "local": {"label": "Local model", "protocol": "chat_completions", "base_url": "http://127.0.0.1:11434/v1",
              "key_env": "", "model_example": "", "token_parameter": "max_tokens"},
}


def credential_envs(provider: str) -> tuple[str, ...]:
    if provider in {"responses", "chat_completions"}:
        return ("DFT_AGENT_API_KEY", "OPENAI_API_KEY")
    if provider == "glm":
        return ("ZAI_API_KEY", "ZHIPUAI_API_KEY")
    name = PROVIDERS[provider]["key_env"]
    return (name,) if name else ()


def effective_key(provider: str, explicit: str | None = None) -> str | None:
    return explicit or next((os.environ[name] for name in credential_envs(provider) if os.getenv(name)), None)


def model_label(provider: str, model: str = "") -> str:
    return model or PROVIDERS.get(provider, {}).get("label", "Model") + " default"
