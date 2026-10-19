"""Recognizable credentials must stay out of ordinary chat and setup history."""

import re


_SECRET = re.compile(
    r"-----BEGIN (?:[A-Z0-9 ]+ )?PRIVATE KEY-----[\s\S]*?(?:-----END [^-]*PRIVATE KEY-----|\Z)"
    r"|\b(?:sk-[A-Za-z0-9_-]{12,}|xai-[A-Za-z0-9_-]{12,}|"
    r"gh[pousr]_[A-Za-z0-9_]{12,}|github_pat_[A-Za-z0-9_]{12,}|AIza[A-Za-z0-9_-]{20,})\b"
    r"|(?i:\b(?:[A-Za-z0-9]+_)*(?:api[_ -]?key|password|passwd|passphrase|"
    r"private[_ -]?key|access[_ -]?token|refresh[_ -]?token|authorization|otp|totp|"
    r"device[_ -]?code)\b[\"']?\s*(?:[:=]|\bis\b)\s*[\"']?[^\s,;\"']+)"
    r"|(?i:\b(?:otp|totp|device[_ -]?code)\s*(?:[0-9]{6,8}|[A-Za-z0-9]{4}-[A-Za-z0-9]{5})\b)"
    r"|(?:密码|口令|验证码|动态码|设备码|(?:API\s*)?密钥)\s*(?:[:：=]|是)\s*[^\s,;，；。]+"
    r"|(?:验证码|动态码|设备码)\s*(?:[0-9]{6,8}|[A-Za-z0-9]{4}-[A-Za-z0-9]{5})\b"
    r"|https?://[^\s/@:]+:[^\s/@]+@"
    r"|(?i:\bBearer\s+[A-Za-z0-9._~+/-]+=*)"
)
_SETUP_PROSE = re.compile(r"(?i:\s+is\s+(required|optional|needed|missing|unset|unavailable|configured|stored)[.!?]?$)|"
                          r"是\s*(可选的?|必需的?|未设置|已设置)$")


def _sensitive(match):
    # Exempt setup prose only for 'is/是'. Explicit assignments are credentials
    # even when their value happens to be 'required' or another ordinary word.
    return _SETUP_PROSE.search(match.group()) is None


def reject_ordinary_text(text):
    """Reject known secret formats before saving or sending ordinary text."""
    if not isinstance(text, str):
        raise ValueError("Enter text.")
    if any(_sensitive(match) for match in _SECRET.finditer(text)):
        raise ValueError("Use the secure credential field. Do not put keys, passwords or login codes in chat.")


def redact(text):
    return _SECRET.sub(lambda match: "[secret omitted]" if _sensitive(match) else match.group(), str(text))
