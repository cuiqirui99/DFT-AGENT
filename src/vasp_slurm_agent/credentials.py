"""Optional API-key storage in the operating system keychain.

Keys are stored only when the user asks. The `keyring` package talks to the
macOS Keychain, Windows Credential Manager or a Linux Secret Service.
"""

from __future__ import annotations

from .providers import PROVIDERS
from .settings import load_settings, save_settings

SERVICE = "dft-agent"
_SYSTEM_BACKENDS = {
    ("keyring.backends.macOS", "Keyring"),
    ("keyring.backends.Windows", "WinVaultKeyring"),
    ("keyring.backends.SecretService", "Keyring"),
    ("keyring.backends.libsecret", "Keyring"),
    ("keyring.backends.kwallet", "DBusKeyring"),
    ("keyring.backends.kwallet", "DBusKeyringKWallet4"),
}


class CredentialError(RuntimeError):
    """The keychain is unavailable or refused the request."""


def _keyring():
    try:
        import keyring
        from keyring import errors  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on the installed extras
        raise CredentialError("Install the keyring package to remember API keys: pip install keyring") from exc
    try:
        backend = keyring.get_keyring()
        identity = lambda item: (type(item).__module__, type(item).__name__)
        backends = list(backend.backends) if identity(backend) == (
            "keyring.backends.chainer", "ChainerBackend") else [backend]
    except Exception:
        raise CredentialError("The system keychain could not be opened.") from None
    if not backends or any(identity(item) not in _SYSTEM_BACKENDS for item in backends):
        raise CredentialError("Use a system keychain: macOS Keychain, Windows Credential Manager, Secret Service or KWallet. File-based keyrings are not supported.")
    # Return the checked native backend, not the module or a dynamic chain that
    # could choose another store on a later read or write.
    return backends[0]


def keyring_available() -> bool:
    try:
        _keyring()
    except CredentialError:
        return False
    return True


def _check_provider(provider: str) -> None:
    if provider not in PROVIDERS or provider == "codex":
        raise CredentialError("Choose an API provider.")


def load_key(provider: str) -> str | None:
    _check_provider(provider)
    try:
        value = _keyring().get_password(SERVICE, provider)
    except CredentialError:
        raise
    except Exception as exc:  # keyring backends raise their own error types
        raise CredentialError("Cannot read the system keychain.") from None
    return value or None


def save_key(provider: str, key: str) -> None:
    _check_provider(provider)
    key = (key or "").strip()
    if not key or any(c in key for c in "\n\r\x00"):
        raise CredentialError("Enter an API key on a single line.")
    try:
        _keyring().set_password(SERVICE, provider, key)
    except CredentialError:
        raise
    except Exception as exc:
        raise CredentialError("Cannot write to the system keychain.") from None
    save_settings({"keychain": True})


def delete_key(provider: str) -> None:
    _check_provider(provider)
    try:
        backend = _keyring()
        if backend.get_password(SERVICE, provider) is None:
            return
        try:
            backend.delete_password(SERVICE, provider)
        except Exception:
            # Some backends report an error for a key another process removed.
            # The exception type alone does not establish that the key is gone.
            if backend.get_password(SERVICE, provider) is not None:
                raise
        if backend.get_password(SERVICE, provider) is not None:
            raise CredentialError("Cannot remove the key: the keychain still contains it.")
    except CredentialError:
        raise
    except Exception as exc:
        raise CredentialError("Cannot remove the key from the system keychain.") from None


def stored_key(provider: str) -> str | None:
    """A remembered key, or None when the user never opted in or nothing is stored."""
    if provider not in PROVIDERS or provider == "codex" or not load_settings().get("keychain"):
        return None
    try:
        return load_key(provider)
    except CredentialError:
        return None
