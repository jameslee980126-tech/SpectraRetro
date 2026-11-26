"""Optional OpenAI-compatible LLM policy shared by browser and live worker."""

from __future__ import annotations

import os
import json
from pathlib import Path

from agentic_retrosynthesis.llm import OpenAICompatibleRanker, OpenAICompatibleStepFallback
from .secret_store import load_api_key


def configured_llm(settings_file=None, credential_file=None):
    settings = {}
    settings_file = Path(settings_file) if settings_file is not None else Path(__file__).resolve().parent / "llm_settings.local.json"
    if settings_file.is_file():
        settings = json.loads(settings_file.read_text(encoding="utf-8-sig"))
        if not isinstance(settings, dict):
            raise ValueError("llm_settings.local.json must contain a JSON object")
    url = os.getenv("LLM_URL", str(settings.get("url", ""))).strip()
    model = os.getenv("LLM_MODEL", str(settings.get("model", ""))).strip()
    if bool(url) != bool(model):
        raise ValueError("LLM_URL and LLM_MODEL must be configured together")
    if not url:
        return None, None
    key = (os.getenv("LLM_API_KEY", "").strip()
           or str(settings.get("api_key") or "").strip()
           or load_api_key(url, path=credential_file))
    raw_temperature = os.getenv("LLM_TEMPERATURE", str(settings.get("temperature", 0)))
    options = dict(timeout=int(os.getenv("LLM_TIMEOUT", str(settings.get("timeout", 45)))),
                   max_tokens=int(os.getenv("LLM_MAX_TOKENS", str(settings.get("max_tokens", 1200)))),
                   json_mode=os.getenv("LLM_JSON_MODE", str(int(settings.get("json_mode", True)))) == "1",
                   temperature=None if raw_temperature.lower() in ("none", "null", "omit") else float(raw_temperature),
                   token_parameter=os.getenv("LLM_TOKEN_PARAMETER", settings.get("token_parameter", "max_tokens")))
    ranker = OpenAICompatibleRanker(url, model, key, **options)
    fallback = (OpenAICompatibleStepFallback(url, model, key, **options)
                if os.getenv("LLM_STEP_FALLBACK", str(int(bool(settings.get("step_fallback", False))))) == "1"
                else None)
    return ranker, fallback
