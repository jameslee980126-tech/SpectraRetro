"""Persist an endpoint-bound LLM credential with Windows CurrentUser DPAPI.

Only ciphertext is written to disk. DPAPI uses the current Windows account;
no credential is placed in command-line arguments, logs, or public status data.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
from urllib.parse import urlsplit, urlunsplit


DEFAULT_PATH = Path(__file__).resolve().parent / "data/private/llm_key.dpapi.json"
_LOCK = threading.RLock()
_PROTECTION = "windows_dpapi_current_user"


class CredentialStoreError(RuntimeError):
    """A credential operation failed without exposing secret material."""


def canonical_endpoint(endpoint):
    if not isinstance(endpoint, str) or not 1 <= len(endpoint.strip()) <= 2048:
        raise ValueError("Provide a valid LLM endpoint URL")
    endpoint = endpoint.strip()
    if any(ord(c) <= 32 for c in endpoint):
        raise ValueError("LLM endpoint URLs cannot contain whitespace")
    try:
        parts = urlsplit(endpoint)
        host, port = parts.hostname, parts.port
    except ValueError:
        raise ValueError("Provide a valid LLM endpoint URL") from None
    if (parts.scheme not in ("https", "http") or not host
            or parts.username or parts.password or parts.query or parts.fragment):
        raise ValueError("LLM endpoint must be HTTP(S) without credentials, query, or fragment")
    if parts.scheme == "http" and host not in ("localhost", "127.0.0.1", "::1"):
        raise ValueError("Remote LLM endpoints require HTTPS")
    host = host.lower()
    authority = f"[{host}]" if ":" in host else host
    if port is not None and port != (443 if parts.scheme == "https" else 80):
        authority += f":{port}"
    return urlunsplit((parts.scheme, authority, parts.path or "/", "", ""))


def _checked_key(key):
    if not isinstance(key, str):
        raise ValueError("API key must be text")
    key = key.strip()
    if not 8 <= len(key) <= 4096 or any(not 33 <= ord(c) <= 126 for c in key):
        raise ValueError("API key must contain 8 to 4096 printable non-space ASCII characters")
    return key


class _DataBlob(ctypes.Structure):
    _fields_ = [("cbData", wintypes.DWORD), ("pbData", ctypes.POINTER(ctypes.c_ubyte))]


def _blob(data):
    buffer = ctypes.create_string_buffer(data)
    return _DataBlob(len(data), ctypes.cast(buffer, ctypes.POINTER(ctypes.c_ubyte))), buffer


def _crypt(data, endpoint, decrypt=False):
    if os.name != "nt":
        raise CredentialStoreError("Persistent credentials require Windows CurrentUser DPAPI")
    crypt32 = ctypes.WinDLL("crypt32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    blob_pointer = ctypes.POINTER(_DataBlob)
    crypt32.CryptProtectData.argtypes = [blob_pointer, wintypes.LPCWSTR, blob_pointer,
                                        ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, blob_pointer]
    crypt32.CryptProtectData.restype = wintypes.BOOL
    crypt32.CryptUnprotectData.argtypes = [blob_pointer, ctypes.POINTER(wintypes.LPWSTR), blob_pointer,
                                          ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, blob_pointer]
    crypt32.CryptUnprotectData.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    input_blob, input_buffer = _blob(data)
    entropy_blob, entropy_buffer = _blob(hashlib.sha256(("CHRetroAgent credential v1\0" + endpoint).encode()).digest())
    output = _DataBlob()
    try:
        if decrypt:
            success = crypt32.CryptUnprotectData(ctypes.byref(input_blob), None, ctypes.byref(entropy_blob),
                                                 None, None, 1, ctypes.byref(output))
        else:
            success = crypt32.CryptProtectData(ctypes.byref(input_blob), "CHRetroAgent external LLM credential",
                                               ctypes.byref(entropy_blob), None, None, 1, ctypes.byref(output))
        if not success:
            raise CredentialStoreError(f"Windows credential {'decryption' if decrypt else 'encryption'} failed (code {ctypes.get_last_error()})")
        return ctypes.string_at(output.pbData, output.cbData)
    finally:
        ctypes.memset(input_buffer, 0, len(input_buffer))
        ctypes.memset(entropy_buffer, 0, len(entropy_buffer))
        if output.pbData:
            ctypes.memset(output.pbData, 0, output.cbData)
            kernel32.LocalFree(ctypes.cast(output.pbData, ctypes.c_void_p))


def _read_record(path):
    try:
        if path.stat().st_size > 65536:
            raise CredentialStoreError("The encrypted credential record exceeds the size limit")
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, UnicodeError):
        raise CredentialStoreError("The encrypted credential record cannot be read") from None
    if (not isinstance(record, dict) or record.get("version") != 1
            or record.get("protection") != _PROTECTION
            or not isinstance(record.get("ciphertext"), str)):
        raise CredentialStoreError("The encrypted credential record has an unsupported format")
    return record


def save_api_key(key, endpoint, path=None):
    endpoint = canonical_endpoint(endpoint)
    key = _checked_key(key)
    target = Path(path) if path is not None else DEFAULT_PATH
    payload = json.dumps({"version": 1, "endpoint": endpoint, "api_key": key}, separators=(",", ":")).encode()
    ciphertext = _crypt(payload, endpoint)
    record = {"version": 1, "protection": _PROTECTION, "endpoint": endpoint,
              "provider_host": urlsplit(endpoint).hostname,
              "ciphertext": base64.b64encode(ciphertext).decode("ascii"),
              "saved_at": datetime.now(timezone.utc).isoformat()}
    temporary = None
    with _LOCK:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix=".llm_key_", suffix=".tmp",
                                             dir=target.parent, delete=False) as output:
                temporary = Path(output.name)
                json.dump(record, output, ensure_ascii=False, indent=2)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, target)
            temporary = None
        except OSError:
            raise CredentialStoreError("The encrypted credential could not be saved") from None
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return {"saved": True, "configured": True, "available": True, "storage": _PROTECTION,
            "provider_host": record["provider_host"], "status": "ready"}


def load_api_key(endpoint, path=None):
    endpoint = canonical_endpoint(endpoint)
    target = Path(path) if path is not None else DEFAULT_PATH
    with _LOCK:
        record = _read_record(target)
        if record is None or record.get("endpoint") != endpoint:
            return None
        try:
            ciphertext = base64.b64decode(record["ciphertext"], validate=True)
            if not ciphertext:
                raise ValueError("Empty encrypted value")
            payload = json.loads(_crypt(ciphertext, endpoint, decrypt=True))
            if not isinstance(payload, dict) or payload.get("version") != 1 or payload.get("endpoint") != endpoint:
                raise ValueError("Endpoint binding mismatch")
            return _checked_key(payload.get("api_key"))
        except (ValueError, UnicodeError, TypeError):
            raise CredentialStoreError("The encrypted credential failed validation") from None


def credential_status(endpoint=None, path=None):
    target = Path(path) if path is not None else DEFAULT_PATH
    status = {"configured": False, "available": False, "storage": _PROTECTION,
              "provider_host": None, "status": "not_configured"}
    try:
        record = _read_record(target)
        if record is None:
            return status
        bound_endpoint = canonical_endpoint(record.get("endpoint"))
        requested = canonical_endpoint(endpoint) if endpoint is not None else bound_endpoint
        status.update(configured=True, provider_host=urlsplit(bound_endpoint).hostname)
        if requested != bound_endpoint:
            return {**status, "status": "endpoint_mismatch"}
        available = load_api_key(requested, path=target) is not None
        return {**status, "available": available, "status": "ready" if available else "not_configured"}
    except (ValueError, CredentialStoreError):
        return {**status, "available": False, "status": "unavailable"}


def main():
    parser = argparse.ArgumentParser(description="Save an external LLM API key using Windows CurrentUser DPAPI")
    parser.add_argument("--endpoint", required=True, help="The exact chat-completions endpoint")
    parser.add_argument("--path", default=None, help="Optional alternate encrypted credential file")
    args = parser.parse_args()
    try:
        key = getpass.getpass("LLM API key (hidden): ") if sys.stdin.isatty() else sys.stdin.readline().rstrip("\r\n")
        status = save_api_key(key, args.endpoint, path=args.path)
        print(json.dumps({"saved": status["saved"], "provider_host": status["provider_host"]}))
    except (ValueError, CredentialStoreError, EOFError):
        print(json.dumps({"saved": False, "error": "Credential save failed; check the endpoint, key, and Windows account"}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
