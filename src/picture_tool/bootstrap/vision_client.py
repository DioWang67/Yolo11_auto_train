"""A vision LLM behind an interface, so the bootstrapper never learns whose.

The company calls its models through an API key and a URL. That is all this
needs, so that is all it assumes: no vendor SDK, no vendor-shaped request
hard-coded, and nothing above :class:`VisionLLMClient` that knows which
service answered. Swapping the endpoint is a configuration change.

Written against ``urllib`` from the standard library rather than an HTTP
package. One request shape and one POST do not justify a new dependency in a
project that has been careful about them.

**The key is never written down.** It is read from the environment by name,
sent in a header, and redacted from every error, log line and provenance
record this module produces. A credential that reaches a provenance file is
a credential in git.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from picture_tool.autotrain import AutoTrainError

#: Verdicts the reviewer may return. Anything else is a protocol error.
ACCEPT = "ACCEPT"
RETRY = "RETRY"
REJECT = "REJECT"
VERDICTS = (ACCEPT, RETRY, REJECT)


class VisionClientError(AutoTrainError):
    """Raised when the vision service could not be reached or understood."""


@dataclass(frozen=True)
class VisionVerdict:
    """One adjudication, plus what it cost."""

    verdict: str
    class_name: str = ""
    confidence: float = 0.0
    reason: str = ""
    next_action: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    attempts: int = 1
    raw: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "next_action": self.next_action,
            "usage": dict(self.usage),
            "attempts": self.attempts,
        }


@dataclass(frozen=True)
class VisionRequest:
    """What the reviewer is shown, cheapest useful thing first.

    ``crops`` are the disputed boxes. ``context_image`` is the whole frame and
    is attached only when a crop alone cannot settle the question --- a crop is
    a fraction of the pixels of a full frame, and sending both every time
    doubles the bill to answer questions most of which the crop answers.
    """

    sample_id: str
    prompt: str
    crops: Sequence[bytes] = ()
    context_image: bytes | None = None
    media_type: str = "image/jpeg"


class VisionLLMClient(Protocol):
    """Anything that can look at images and return a verdict."""

    def judge(self, request: VisionRequest) -> VisionVerdict:
        """Adjudicate one sample."""


@dataclass(frozen=True)
class VisionEndpointConfig:
    """Everything about the service, as configuration.

    Only the *names* of the environment variables live here, never their
    values, so this object is safe to serialise, log and commit.
    """

    url_env: str = "VISION_LLM_URL"
    key_env: str = "VISION_LLM_API_KEY"
    model: str = "claude-opus-5"
    path: str = "/v1/messages"
    auth_header: str = "x-api-key"
    auth_prefix: str = ""
    extra_headers: Mapping[str, str] = field(
        default_factory=lambda: {"anthropic-version": "2023-06-01"}
    )
    max_tokens: int = 800
    timeout_seconds: float = 120.0
    #: Retries on transport or protocol failure, not on a verdict of RETRY.
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    #: Which wire dialect to build and parse. "messages" is the
    #: Anthropic-style content-block shape; "chat_completions" the OpenAI one.
    dialect: str = "messages"

    def to_dict(self) -> dict[str, Any]:
        return {
            "url_env": self.url_env,
            "key_env": self.key_env,
            "model": self.model,
            "path": self.path,
            "dialect": self.dialect,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
        }


class HttpVisionLLMClient:
    """Posts images and a question to a configured HTTP endpoint."""

    def __init__(
        self,
        config: VisionEndpointConfig | None = None,
        *,
        environ: Mapping[str, str] | None = None,
        opener: Any = None,
    ) -> None:
        self.config = config or VisionEndpointConfig()
        env = environ if environ is not None else os.environ
        url = str(env.get(self.config.url_env) or "").strip()
        key = str(env.get(self.config.key_env) or "").strip()
        if not url:
            raise VisionClientError(
                f"{self.config.url_env} is not set. The endpoint is configuration; "
                "there is deliberately no default, because a default here would "
                "silently send production images somewhere nobody chose."
            )
        if not key:
            raise VisionClientError(
                f"{self.config.key_env} is not set. Refusing to call an "
                "authenticated endpoint without a credential the caller supplied."
            )
        self._url = url.rstrip("/") + self.config.path
        self._key = key
        self._opener = opener or urllib.request.urlopen

    # -- request ---------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            self.config.auth_header: f"{self.config.auth_prefix}{self._key}",
        }
        headers.update(dict(self.config.extra_headers))
        return headers

    def _body(self, request: VisionRequest) -> dict[str, Any]:
        images = list(request.crops)
        if request.context_image is not None:
            images.append(request.context_image)
        if self.config.dialect == "chat_completions":
            content: list[dict[str, Any]] = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{request.media_type};base64,"
                        f"{base64.b64encode(blob).decode('ascii')}"
                    },
                }
                for blob in images
            ]
            content.append({"type": "text", "text": request.prompt})
            return {
                "model": self.config.model,
                "max_tokens": self.config.max_tokens,
                "messages": [{"role": "user", "content": content}],
            }
        blocks: list[dict[str, Any]] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": request.media_type,
                    "data": base64.b64encode(blob).decode("ascii"),
                },
            }
            for blob in images
        ]
        blocks.append({"type": "text", "text": request.prompt})
        return {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            "messages": [{"role": "user", "content": blocks}],
        }

    def judge(self, request: VisionRequest) -> VisionVerdict:
        payload = json.dumps(self._body(request)).encode("utf-8")
        last_error = ""
        for attempt in range(1, self.config.max_retries + 2):
            try:
                raw = self._post(payload)
            except VisionClientError as exc:
                last_error = _redact(str(exc), self._key)
                if attempt > self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff_seconds * attempt)
                continue
            try:
                return _to_verdict(raw, attempts=attempt)
            except VisionClientError as exc:
                # A malformed answer is worth one more try: these services
                # occasionally wrap the JSON in prose. It is not worth many.
                last_error = _redact(str(exc), self._key)
                if attempt > self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff_seconds * attempt)
        raise VisionClientError(
            f"{request.sample_id}: no usable answer after "
            f"{self.config.max_retries + 1} attempt(s). {last_error}"
        )

    def _post(self, payload: bytes) -> dict[str, Any]:
        req = urllib.request.Request(
            self._url, data=payload, headers=self._headers(), method="POST"
        )
        try:
            with self._opener(req, timeout=self.config.timeout_seconds) as response:
                body = response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise VisionClientError(
                f"HTTP {exc.code} from the vision endpoint: "
                f"{_redact(detail, self._key)}"
            ) from None
        except (urllib.error.URLError, OSError, TimeoutError) as exc:
            raise VisionClientError(
                f"Could not reach the vision endpoint: {_redact(str(exc), self._key)}"
            ) from None
        try:
            parsed = json.loads(body)
        except ValueError:
            raise VisionClientError(
                f"Response was not JSON: {_redact(body[:400], self._key)}"
            ) from None
        if not isinstance(parsed, dict):
            raise VisionClientError("Response JSON was not an object.")
        return parsed


# ---------------------------------------------------------------------------
# Adapters


def extract_text(payload: Mapping[str, Any]) -> str:
    """Pull the assistant's text out of whichever shape came back.

    Three dialects are tried and then the search gives up rather than
    guessing: a wrong guess here would read some other field as a verdict.
    """
    content = payload.get("content")
    if isinstance(content, list):
        parts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") in (None, "text")
        ]
        if any(parts):
            return "\n".join(parts)
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if isinstance(message, dict):
            text = message.get("content")
            if isinstance(text, str):
                return text
            if isinstance(text, list):
                return "\n".join(
                    str(b.get("text", "")) for b in text if isinstance(b, dict)
                )
    for key in ("output_text", "text", "result", "response"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    raise VisionClientError(
        "Could not find assistant text in the response. Keys present: "
        + ", ".join(sorted(payload)[:12])
    )


def extract_usage(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Token accounting, if the service reports any."""
    usage = payload.get("usage")
    if isinstance(usage, dict):
        return {str(k): v for k, v in usage.items()}
    return {}


def parse_verdict(text: str) -> dict[str, Any]:
    """Read the JSON object out of the reply, tolerating prose around it."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise VisionClientError(f"No JSON object in the reply: {stripped[:200]}")
    try:
        parsed = json.loads(stripped[start : end + 1])
    except ValueError as exc:
        raise VisionClientError(f"Reply JSON did not parse: {exc}") from None
    if not isinstance(parsed, dict):
        raise VisionClientError("Reply JSON was not an object.")
    return parsed


def _to_verdict(payload: Mapping[str, Any], *, attempts: int) -> VisionVerdict:
    parsed = parse_verdict(extract_text(payload))
    verdict = str(parsed.get("verdict", "")).strip().upper()
    if verdict not in VERDICTS:
        raise VisionClientError(
            f"{verdict!r} is not one of {', '.join(VERDICTS)}."
        )
    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return VisionVerdict(
        verdict=verdict,
        class_name=str(parsed.get("class", parsed.get("class_name", "")) or ""),
        confidence=max(0.0, min(1.0, confidence)),
        reason=str(parsed.get("reason", "") or ""),
        next_action=str(parsed.get("next_action", "") or ""),
        usage=extract_usage(payload),
        attempts=attempts,
        raw=parsed,
    )


def _redact(text: str, secret: str) -> str:
    """Never let the credential out, not even inside an error message."""
    if secret and secret in text:
        text = text.replace(secret, "<redacted>")
    return text


def crop_bytes(image_path: str | Path, box: Any, *, pad: float = 0.15) -> bytes:
    """Encode one disputed box as JPEG, with a little context around it.

    Padded because a box cropped exactly to its own edges loses what makes a
    colour readable --- the neighbouring wires a person compares it against.
    """
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise VisionClientError(f"Could not read {image_path}")
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box.to_pixels(width, height)
    dx = int((x2 - x1) * pad)
    dy = int((y2 - y1) * pad)
    crop = image[
        max(0, y1 - dy) : min(height, y2 + dy),
        max(0, x1 - dx) : min(width, x2 + dx),
    ]
    if crop.size == 0:
        raise VisionClientError("Cropped region is empty")
    ok, buffer = cv2.imencode(".jpg", crop)
    if not ok:
        raise VisionClientError("Could not encode the crop")
    return bytes(buffer)


def whole_image_bytes(image_path: str | Path, *, max_side: int = 1024) -> bytes:
    """The full frame, downscaled --- context does not need full resolution."""
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        raise VisionClientError(f"Could not read {image_path}")
    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
    ok, buffer = cv2.imencode(".jpg", image)
    if not ok:
        raise VisionClientError("Could not encode the image")
    return bytes(buffer)
