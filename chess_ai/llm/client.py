#!/usr/bin/env python3
"""Pure Ollama API wrapper for Chess AI."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Tuple

# Configuration via environment variables
MODEL = os.getenv("OLLAMA_MODEL", "llama4:latest")
OLLAMA_CHAT_URL = os.getenv("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))
DEFAULT_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.5"))

def call_ollama(system_prompt: str, user_prompt: str, temperature: float) -> Tuple[str, str]:
    """Call Ollama `/api/chat` and return `(message.content, raw_json_response)`.

    Args:
        system_prompt: The system instructions.
        user_prompt: The user query.
        temperature: Sampling temperature.

    Returns:
        A tuple of (extracted content, raw response text).

    Raises:
        ValueError: If response is missing `message.content`.
        urllib.error.URLError: If network request fails.
    """
    body = {
        "model": MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "options": {"temperature": temperature},
    }

    req = urllib.request.Request(
        OLLAMA_CHAT_URL,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
        raw = response.read().decode("utf-8")

    data = json.loads(raw)
    content = data.get("message", {}).get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("Ollama response missing `message.content`.")

    return content.strip(), raw
