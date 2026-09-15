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
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from picture_tool.autotrain import AutoTrainError

#: Every outbound call is refused unless this variable holds :data:`CALLS_ALLOWED`.
#:
#: The endpoint being configuration is not by itself a safeguard. A config
#: file gets copied, an environment gets inherited, and the station that ends
#: up calling a language model with production frames is the one nobody meant
#: to. So the default is refusal and enabling it is a deliberate act, in the
#: same spirit as ``trainer.assert_no_forbidden_tasks``: the guarantee must
#: not depend on some YAML staying correct.
#:
#: Checked at the moment of the request rather than at construction, so no
#: caller -- including one written later -- can arrange to skip it.
CALLS_ENABLED_ENV = "PICTURE_TOOL_VISION_CALLS"

#: Not "1" or "true". A deployment that sets flags wholesale does not set
#: this by accident, and the word says what is being permitted.
CALLS_ALLOWED = "allow"

#: Verdicts the reviewer may return. Anything else is a protocol error.
ACCEPT = "ACCEPT"
RETRY = "RETRY"
REJECT = "REJECT"
VERDICTS = (ACCEPT, RETRY, REJECT)


class VisionClientError(AutoTrainError):
    """Raised when the vision service could not be reached or understood."""


class VisionCallsDisabledError(AutoTrainError):
    """Raised when outbound calls have not been enabled.

    Deliberately *not* a :class:`VisionClientError`. That class is what the
    retry loop catches and what a caller records as a failed sample, and this
    is neither: retrying cannot help, and every remaining sample would fail
    identically. It has to stop the run, not fill a report with 122 copies of
    the same refusal.
    """


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
    #: Which model answered. Two reviewers disagreeing is a finding; two
    #: reviewers averaged together because the record did not say which was
    #: which is a corrupted dataset, and the file that holds these already
    #: contains verdicts from more than one.
    model: str = ""
    #: True when the reply was not valid JSON and the fields were recovered
    #: by :func:`parse_verdict`. Carried all the way to the report: a
    #: salvaged ``reason`` is a reconstruction, and a reader deciding what to
    #: trust needs to be told which ones they are.
    repaired: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 4),
            "reason": self.reason,
            "next_action": self.next_action,
            "usage": dict(self.usage),
            "attempts": self.attempts,
            "model": self.model,
            "repaired": self.repaired,
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
    #: Base deadline. The endpoint buffers the whole reply before sending it,
    #: so this is in practice a generation deadline rather than a stall
    #: detector.
    timeout_seconds: float = 120.0
    #: Added per attached image, because generation time grows with the
    #: number of images and a flat deadline therefore fails the *largest*
    #: samples first. The 213-sample run aborted on a sample carrying two
    #: crops and the whole frame; at three images a flat 120s was short.
    timeout_per_image_seconds: float = 45.0
    #: Retries on transport or protocol failure, not on a verdict of RETRY.
    max_retries: int = 2
    retry_backoff_seconds: float = 2.0
    #: Which wire dialect to build and parse. "messages" is the
    #: Anthropic-style content-block shape; "chat_completions" the OpenAI one.
    dialect: str = "messages"
    #: Whether this endpoint authenticates at all. Unauthenticated services do
    #: exist on the internal network, and the honest way to talk to one is to
    #: say so here -- not to satisfy the credential check with a placeholder,
    #: which would also silence it for endpoints that really do want a key.
    #: Recorded in :meth:`to_dict`, so provenance shows which runs went to an
    #: endpoint with no access control in front of it.
    requires_credential: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "url_env": self.url_env,
            "key_env": self.key_env,
            "model": self.model,
            "path": self.path,
            "dialect": self.dialect,
            "max_tokens": self.max_tokens,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "timeout_per_image_seconds": self.timeout_per_image_seconds,
            "requires_credential": self.requires_credential,
        }


def openai_compatible_profile(
    *,
    model: str,
    url_env: str,
    key_env: str = "",
    requires_credential: bool = False,
    **overrides: Any,
) -> VisionEndpointConfig:
    """The shape of an OpenAI-compatible server, without naming one.

    Local servers -- llama.cpp, vLLM, Ollama -- speak the ``/chat/completions``
    dialect and frequently sit behind no authentication at all. This captures
    the *shape*: dialect, path, and the absence of Anthropic's headers.

    The address is deliberately not a parameter with a default. An endpoint
    baked into this file would be one copy away from a station pointing at it,
    which is exactly what :data:`CALLS_ENABLED_ENV` exists to prevent. The
    caller names an environment variable; somebody chooses its value.

    Such servers are usually slower per token than a hosted one but answer
    far sooner on image-heavy requests, so the per-image deadline is lowered
    rather than inherited.
    """
    settings: dict[str, Any] = {
        "url_env": url_env,
        "key_env": key_env or "UNUSED_NO_CREDENTIAL",
        "model": model,
        "path": "/chat/completions",
        "dialect": "chat_completions",
        "auth_header": "authorization",
        "auth_prefix": "Bearer ",
        "extra_headers": {},
        "requires_credential": requires_credential,
        "timeout_seconds": 120.0,
        "timeout_per_image_seconds": 60.0,
        # Reasoning models spend this budget thinking *before* they write the
        # answer, and the answer is what gets truncated when it runs out. At
        # 800 -- fine for a hosted non-reasoning model -- two of the first
        # three real samples came back with reasoning and an empty reply.
        "max_tokens": 2048,
    }
    settings.update(overrides)
    return VisionEndpointConfig(**settings)


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
        if not key and self.config.requires_credential:
            raise VisionClientError(
                f"{self.config.key_env} is not set. Refusing to call an "
                "authenticated endpoint without a credential the caller "
                "supplied. If the endpoint genuinely has no authentication, "
                "say so with requires_credential=False rather than supplying "
                "a placeholder -- the difference is recorded in provenance."
            )
        self._url = url.rstrip("/") + self.config.path
        # An endpoint declared keyless is sent no credential at all, whatever
        # happens to be in the environment. Without this, pointing a run at a
        # local unauthenticated server while a hosted provider's key is still
        # exported would put that key on the wire as a Bearer token -- to a
        # different operator, over plaintext HTTP. Declaring the endpoint
        # keyless has to mean no key leaves the process.
        self._key = "" if not self.config.requires_credential else key
        self._environ = env
        self._opener = opener or urllib.request.urlopen

    # -- request ---------------------------------------------------------

    def _assert_calls_allowed(self) -> None:
        """Refuse to reach the network unless somebody said to.

        Deliberately re-read from the environment on every request instead of
        being cached at construction: a long-lived process should not carry a
        permission it was granted once.
        """
        if str(self._environ.get(CALLS_ENABLED_ENV, "")).strip() == CALLS_ALLOWED:
            return
        raise VisionCallsDisabledError(
            f"Refusing to call {self.config.model} at the configured endpoint: "
            f"{CALLS_ENABLED_ENV} is not set to '{CALLS_ALLOWED}'. This guard "
            "is default-deny on purpose, so that deployed and automated runs "
            "cannot send production frames to a language model by inheriting "
            "an environment. Set it for the one command that should spend "
            "tokens, not in a profile or a service definition."
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
        }
        if self._key:
            headers[self.config.auth_header] = (
                f"{self.config.auth_prefix}{self._key}"
            )
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

    def request_timeout(self, image_count: int) -> float:
        """The deadline this request is allowed, given how much it carries."""
        return self.config.timeout_seconds + (
            self.config.timeout_per_image_seconds * max(0, image_count)
        )

    def judge(self, request: VisionRequest) -> VisionVerdict:
        payload = json.dumps(self._body(request)).encode("utf-8")
        images = len(request.crops) + (request.context_image is not None)
        timeout = self.request_timeout(images)
        last_error = ""
        for attempt in range(1, self.config.max_retries + 2):
            try:
                raw = self._post(payload, timeout)
            except VisionClientError as exc:
                last_error = _redact(str(exc), self._key)
                if attempt > self.config.max_retries:
                    break
                time.sleep(self.config.retry_backoff_seconds * attempt)
                continue
            try:
                return _to_verdict(
                    raw, attempts=attempt, fallback_model=self.config.model
                )
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

    def _post(self, payload: bytes, timeout: float | None = None) -> dict[str, Any]:
        # Here rather than in judge(): this is the only place bytes leave the
        # process, so a caller cannot arrange to miss it.
        self._assert_calls_allowed()
        req = urllib.request.Request(
            self._url, data=payload, headers=self._headers(), method="POST"
        )
        deadline = self.config.timeout_seconds if timeout is None else timeout
        try:
            with self._opener(req, timeout=deadline) as response:
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
            # An empty string is not an answer. A reasoning model that ran out
            # of budget returns exactly that, and treating it as text turns a
            # truncation into a baffling "no JSON object" further down.
            if isinstance(text, str) and text.strip():
                return text
            if isinstance(text, list):
                return "\n".join(
                    str(b.get("text", "")) for b in text if isinstance(b, dict)
                )
    for key in ("output_text", "text", "result", "response"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value
    if was_truncated(payload):
        raise VisionClientError(
            "The reply was cut off at max_tokens before any answer was "
            "written. A reasoning model spends that budget thinking first, "
            "so the answer is what goes missing --- raise max_tokens rather "
            "than retrying, which will truncate identically."
        )
    raise VisionClientError(
        "Could not find assistant text in the response. Keys present: "
        + ", ".join(sorted(payload)[:12])
    )


def was_truncated(payload: Mapping[str, Any]) -> bool:
    """Whether the service stopped because it ran out of token budget."""
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        if choices[0].get("finish_reason") == "length":
            return True
    return payload.get("stop_reason") == "max_tokens"


#: OpenAI-style usage names mapped onto the ones the reports already total.
#: Without this an OpenAI-compatible server reports its cost into fields
#: nothing reads, and the run appears to have been free.
_USAGE_ALIASES = {
    "prompt_tokens": "input_tokens",
    "completion_tokens": "output_tokens",
}


def extract_usage(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Token accounting, if the service reports any.

    Both spellings are kept: the canonical one so totals add up across
    endpoints, and the original so the record still says what the service
    actually returned.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return {}
    normalised = {str(k): v for k, v in usage.items()}
    for source, canonical in _USAGE_ALIASES.items():
        if source in normalised and canonical not in normalised:
            normalised[canonical] = normalised[source]
    return normalised


#: The keys the reviewer is asked to return. Salvage recognises these and
#: nothing else --- a repair free to invent fields would turn a malformed
#: reply into one that merely looks answered.
_REPLY_KEYS: tuple[str, ...] = (
    "verdict",
    "class",
    "class_name",
    "confidence",
    "reason",
    "next_action",
)

_REPLY_KEY_RE = re.compile(
    '"(' + "|".join(re.escape(key) for key in _REPLY_KEYS) + r')"\s*:\s*'
)

#: A value ends where the enclosing object's own punctuation begins.
_TRAILING_PUNCTUATION_RE = re.compile(r"[\s,}]+$")

#: The escapes a reply that failed strict parsing may still carry.
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\", "/": "/"}


@dataclass(frozen=True)
class ParsedReply:
    """The reviewer's fields, and whether they survived strict JSON."""

    fields: Mapping[str, Any]
    repaired: bool = False


def parse_verdict(text: str) -> ParsedReply:
    """Read the JSON object out of the reply, tolerating prose around it.

    Most replies are valid JSON and are parsed as such. When one is not, the
    cause seen in practice is an unescaped quote inside ``reason`` --- the
    reviewer quoting the label it is arguing against --- which ends the
    string early and makes the remainder unreadable to :func:`json.loads`.
    Two of the 213 samples in the first full run were lost exactly that way,
    and a lost sample still cost the tokens it spent.

    A failed parse therefore falls back to :func:`_salvage_reply`, which does
    not guess at structure: the prompt fixes the set of keys, so each value
    can be taken to run from its own colon to the next key, whatever quoting
    happens in between. The result is marked ``repaired``, and stays marked
    all the way into the report --- a salvaged ``reason`` is a
    reconstruction, and the reader deciding what to trust has to be told so.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise VisionClientError(f"No JSON object in the reply: {stripped[:200]}")
    blob = stripped[start : end + 1]
    try:
        parsed = json.loads(blob)
    except ValueError as exc:
        return ParsedReply(fields=_salvage_reply(blob, str(exc)), repaired=True)
    if not isinstance(parsed, dict):
        raise VisionClientError("Reply JSON was not an object.")
    return ParsedReply(fields=parsed)


def _salvage_reply(blob: str, parse_error: str) -> dict[str, Any]:
    """Recover the declared fields from a reply that is not valid JSON.

    Refuses to return anything without a ``verdict``: a salvage that yielded
    only a reason would hand the caller a confident-looking record with no
    decision in it.
    """
    matches = list(_REPLY_KEY_RE.finditer(blob))
    fields: dict[str, Any] = {}
    for index, match in enumerate(matches):
        stop = matches[index + 1].start() if index + 1 < len(matches) else len(blob)
        fields[match.group(1)] = _coerce_value(blob[match.end() : stop])
    if "verdict" not in fields:
        raise VisionClientError(
            f"Reply JSON did not parse ({parse_error}) and no verdict could "
            f"be recovered from it: {blob[:200]}"
        )
    return fields


def _coerce_value(raw: str) -> Any:
    """One value, read without trusting the quoting inside it."""
    text = _TRAILING_PUNCTUATION_RE.sub("", raw.strip())
    if text.startswith('"'):
        closing = text.rfind('"')
        return _unescape(text[1:closing] if closing > 0 else text[1:])
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "none", ""):
        return None
    try:
        return float(text)
    except ValueError:
        return text


def _unescape(text: str) -> str:
    """Undo the escapes a half-valid reply may still carry."""
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            out.append(_ESCAPES.get(text[index + 1], "\\" + text[index + 1]))
            index += 2
            continue
        out.append(char)
        index += 1
    return "".join(out)


def _to_verdict(
    payload: Mapping[str, Any], *, attempts: int, fallback_model: str = ""
) -> VisionVerdict:
    # Prefer what the service says answered over what was asked for: a
    # gateway is free to route the request elsewhere, and the record should
    # name the model that actually spoke.
    reply = parse_verdict(extract_text(payload))
    parsed = reply.fields
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
        model=str(payload.get("model") or fallback_model or ""),
        repaired=reply.repaired,
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
