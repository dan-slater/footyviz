"""Optional, swappable LLM/VLM enrichment layer.

The football-vision pipeline runs 100% without this module — detection,
tracking, team colours and whistle detection are all pure CV. This layer is an
*optional* enhancement gated by the `llm` block in configs/veo.yaml
(`enabled: false` by default).

Usage pattern in callers (never a hard dependency):

    from .llm import get_llm, uses
    llm = get_llm(cfg)                     # None when llm.enabled is false
    if llm and uses(cfg, "scoreboard"):
        score = llm.read_scoreboard(crop)
    else:
        ...                                # pure-CV fallback / skip

`get_llm` returns a ready `LLMClient` or `None` — never a broken object — so the
`if llm and uses(...)` guard is all a caller needs.

Backends (selected by `llm.provider`): GeminiClient (primary, google-genai),
OpenAIClient (openai SDK), LocalClient (Ollama). All share one interface and
return plain dicts/str so the rest of the pipeline stays provider-agnostic.
"""
from __future__ import annotations

import abc
import base64
import json
import os
import re
from typing import Any

import numpy as np

# An "image" argument may be a BGR ndarray (as OpenCV produces), a path to an
# image file, or already-encoded PNG/JPEG bytes.
ImageLike = "np.ndarray | str | bytes"

_REPO_ENV = os.path.join(os.path.dirname(__file__), "..", "..", ".env")


def _load_repo_env(path: str = _REPO_ENV) -> None:
    """Fill unset env vars from this repo's own gitignored .env (KEY=VALUE lines).

    Keys come only from here or the caller's environment — never from another
    project's .env."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.strip().strip('"').strip("'")
            if v and not os.environ.get(k.strip()):
                os.environ[k.strip()] = v


_DEFAULT_MODELS = {
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "local": "qwen2.5-vl",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _encode_png(img: Any) -> bytes:
    """Normalise any ImageLike into PNG bytes."""
    if isinstance(img, bytes):
        return img
    if isinstance(img, str):
        with open(img, "rb") as fh:
            return fh.read()
    if isinstance(img, np.ndarray):
        import cv2

        ok, buf = cv2.imencode(".png", img)
        if not ok:
            raise ValueError("failed to PNG-encode ndarray")
        return buf.tobytes()
    raise TypeError(f"unsupported image type: {type(img)}")


def _parse_json(text: str) -> dict:
    """Defensively parse JSON that a model may wrap in prose or ``` fences."""
    if not text:
        return {}
    t = text.strip()
    # strip markdown code fences
    fence = re.search(r"```(?:json)?\s*(.*?)\s*```", t, re.DOTALL)
    if fence:
        t = fence.group(1).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # fall back to the first balanced {...} span
    start = t.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(t)):
            if t[i] == "{":
                depth += 1
            elif t[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start : i + 1])
                    except json.JSONDecodeError:
                        break
    return {"_unparsed": text}


def uses(config: dict, feature: str) -> bool:
    """True iff llm.enabled AND llm.uses[feature] are both set in the config."""
    llm = config.get("llm", config) if isinstance(config, dict) else {}
    if not llm.get("enabled"):
        return False
    return bool(llm.get("uses", {}).get(feature, False))


# --------------------------------------------------------------------------- #
# interface
# --------------------------------------------------------------------------- #
# Compact prompts; every structured method asks for strict JSON.
_P_PLAYER = (
    "You are labelling a soccer player from one or more cropped images of the "
    "same person. Return ONLY JSON with keys: hair, skin, shirt, shorts, socks, "
    "shoes, build. Each value is a short string (e.g. shirt:'white v-neck'). "
    "Use 'unknown' if not visible."
)
_P_EVENT = (
    "These frames are from a 5-a-side match around a referee whistle. Decide if "
    "a goal was just scored. Return ONLY JSON: "
    '{"is_goal": bool, "team": "<colour/bib or unknown>", "note": "<short>"}.'
)
_P_SCORE = (
    "This is a crop of a small illuminated LED/number scoreboard, possibly "
    "overexposed. Read the two scores. Return ONLY JSON: "
    '{"score_home": int, "score_away": int, "confidence": 0.0-1.0} '
    'or {"readable": false} if you cannot read digits.'
)


class LLMClient(abc.ABC):
    """Provider-agnostic interface the pipeline codes against."""

    def __init__(self, model: str):
        self.model = model

    @abc.abstractmethod
    def _vision(self, prompt: str, images: list, want_json: bool) -> str:
        """Send a prompt + images, return raw text."""

    # -- high-level typed calls -------------------------------------------- #
    def describe_player(self, crops: list) -> dict:
        """Rich appearance fingerprint from crops of one player."""
        return _parse_json(self._vision(_P_PLAYER, list(crops), want_json=True))

    def classify_event(self, frames: list, context: str = "") -> dict:
        prompt = _P_EVENT + (f"\nContext: {context}" if context else "")
        return _parse_json(self._vision(prompt, list(frames), want_json=True))

    def read_scoreboard(self, crop) -> dict:
        return _parse_json(self._vision(_P_SCORE, [crop], want_json=True))

    def write_report(self, stats: dict, events: list) -> str:
        prompt = (
            "Write a short, punchy match summary (<=150 words) for an amateur "
            "5-a-side game from these stats and events. Plain prose, no preamble.\n"
            f"STATS: {json.dumps(stats, default=str)}\n"
            f"EVENTS: {json.dumps(events, default=str)}"
        )
        return self._vision(prompt, [], want_json=False).strip()


# --------------------------------------------------------------------------- #
# backends
# --------------------------------------------------------------------------- #
class GeminiClient(LLMClient):
    """Primary backend — google-genai SDK, key from $GEMINI_API_KEY."""

    def __init__(self, model: str):
        super().__init__(model)
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Export it before enabling the Gemini "
                "backend, e.g.:\n"
                "  export GEMINI_API_KEY=...\n"
                "or put it in this repo's gitignored .env (see .env.example)."
            )
        try:
            from google import genai
        except ImportError as e:
            raise RuntimeError(
                "google-genai not installed. Run: "
                "~/.local/bin/uv pip install google-genai"
            ) from e
        self._genai = genai
        self._client = genai.Client(api_key=key)

    def _vision(self, prompt: str, images: list, want_json: bool) -> str:
        from google.genai import types

        contents: list = [prompt]
        for img in images:
            contents.append(
                types.Part.from_bytes(data=_encode_png(img), mime_type="image/png")
            )
        cfg = types.GenerateContentConfig(
            response_mime_type="application/json" if want_json else "text/plain"
        )
        resp = self._client.models.generate_content(
            model=self.model, contents=contents, config=cfg
        )
        return resp.text or ""


class OpenAIClient(LLMClient):
    """OpenAI backend — openai SDK, key from $OPENAI_API_KEY."""

    def __init__(self, model: str):
        super().__init__(model)
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Export it before enabling the OpenAI "
                "backend, or put it in this repo's gitignored .env (see .env.example)."
            )
        try:
            from openai import OpenAI
        except ImportError as e:
            raise RuntimeError(
                "openai not installed. Run: ~/.local/bin/uv pip install openai"
            ) from e
        self._client = OpenAI(api_key=key)

    def _vision(self, prompt: str, images: list, want_json: bool) -> str:
        content: list = [{"type": "text", "text": prompt}]
        for img in images:
            b64 = base64.b64encode(_encode_png(img)).decode()
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                }
            )
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
        }
        if want_json:
            kwargs["response_format"] = {"type": "json_object"}
        resp = self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


class LocalClient(LLMClient):
    """Local backend — Ollama vision model at http://localhost:11434."""

    def __init__(self, model: str, host: str = "http://localhost:11434"):
        super().__init__(model)
        self.host = host.rstrip("/")

    def _vision(self, prompt: str, images: list, want_json: bool) -> str:
        try:
            import urllib.request
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("stdlib urllib unavailable") from e
        payload = {
            "model": self.model,
            "prompt": prompt,
            "images": [base64.b64encode(_encode_png(i)).decode() for i in images],
            "stream": False,
        }
        if want_json:
            payload["format"] = "json"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.host}/api/generate", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read().decode()).get("response", "")
        except Exception as e:  # Ollama not running / model missing
            raise RuntimeError(
                f"Local Ollama call failed ({e}). Is `ollama serve` running with a "
                f"vision model pulled (e.g. `ollama pull {self.model}`)?"
            ) from e


_BACKENDS = {"gemini": GeminiClient, "openai": OpenAIClient, "local": LocalClient}


# --------------------------------------------------------------------------- #
# factory
# --------------------------------------------------------------------------- #
def get_llm(config: dict) -> LLMClient | None:
    """Build the configured LLM client, or None when the layer is disabled.

    Accepts either the full pipeline config (reads its ``llm`` block) or an
    ``llm`` block directly. Returns None when ``enabled`` is falsy so callers
    can simply ``if llm and uses(cfg, feature): ...``.
    """
    llm = config.get("llm", config) if isinstance(config, dict) else {}
    if not llm.get("enabled"):
        return None
    provider = (llm.get("provider") or "gemini").lower()
    if provider not in _BACKENDS:
        raise ValueError(
            f"unknown llm.provider {provider!r}; expected one of {list(_BACKENDS)}"
        )
    model = llm.get("model") or _DEFAULT_MODELS[provider]
    _load_repo_env()
    return _BACKENDS[provider](model)
