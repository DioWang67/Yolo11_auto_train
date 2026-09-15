"""The two ways the first full review run lost samples it had paid for.

The load-bearing tests are the two that reproduce those losses: a reply whose
``reason`` carries an unescaped quote, and a request carrying enough images
that a flat deadline expires before the answer arrives. Both cost real tokens
on 2026-09-14 and returned nothing.

Everything else here guards a property that was previously untested entirely:
this module had no tests at all, so the credential redaction and the refusal
to call an endpoint without one were only as good as their source.
"""

from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from picture_tool.bootstrap.vision_client import (
    CALLS_ALLOWED,
    CALLS_ENABLED_ENV,
    HttpVisionLLMClient,
    VisionCallsDisabledError,
    VisionClientError,
    VisionEndpointConfig,
    VisionRequest,
    extract_usage,
    openai_compatible_profile,
    parse_verdict,
)

KEY = "sk-not-a-real-key-0123456789"
ENVIRON = {
    "VISION_LLM_URL": "http://endpoint.invalid",
    "VISION_LLM_API_KEY": KEY,
    CALLS_ENABLED_ENV: CALLS_ALLOWED,
}


class _FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class _RecordingOpener:
    """Answers with a canned body and remembers how it was called."""

    def __init__(self, body: str) -> None:
        self.body = body
        self.timeouts: list[float] = []

    def __call__(self, request: Any, timeout: float | None = None) -> _FakeResponse:
        self.timeouts.append(float(timeout or 0.0))
        return _FakeResponse(self.body)


class _RaisingOpener:
    def __init__(self, error: BaseException) -> None:
        self.error = error

    def __call__(self, request: Any, timeout: float | None = None) -> _FakeResponse:
        raise self.error


def _reply(text: str) -> str:
    """One assistant reply in the content-block dialect."""
    return json.dumps(
        {
            "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
    )


def _client(opener: Any, **config: Any) -> HttpVisionLLMClient:
    return HttpVisionLLMClient(
        VisionEndpointConfig(**config), environ=ENVIRON, opener=opener
    )


# -- the reply that could not be parsed -------------------------------------

#: The shape that failed twice in the 213-sample run: the reviewer quotes the
#: label it is arguing against, inside a JSON string, without escaping it.
UNESCAPED_QUOTE_REPLY = (
    '{"verdict": "REJECT", "class": "Black", "confidence": 0.92, '
    '"reason": "The crop shows a matte black wire on a green PCB (the '
    'detector\'s "Green" call is wrong)", "next_action": "relabel as Black"}'
)


def test_the_unescaped_quote_reply_really_is_invalid_json() -> None:
    """Guards the premise: if this ever parses, the salvage test is vacuous."""
    with pytest.raises(ValueError):
        json.loads(UNESCAPED_QUOTE_REPLY)


def test_an_unescaped_quote_in_reason_is_recovered_not_lost() -> None:
    reply = parse_verdict(UNESCAPED_QUOTE_REPLY)

    assert reply.repaired is True
    assert reply.fields["verdict"] == "REJECT"
    assert reply.fields["class"] == "Black"
    assert reply.fields["confidence"] == pytest.approx(0.92)
    assert reply.fields["next_action"] == "relabel as Black"
    # The quotes that broke the parse survive inside the recovered text.
    assert '"Green"' in reply.fields["reason"]
    assert reply.fields["reason"].endswith("call is wrong)")


def test_a_repaired_reply_is_marked_all_the_way_into_the_record() -> None:
    """A reconstruction must never be indistinguishable from a clean answer."""
    client = _client(_RecordingOpener(_reply(UNESCAPED_QUOTE_REPLY)))

    verdict = client.judge(VisionRequest(sample_id="s", prompt="p"))

    assert verdict.repaired is True
    assert verdict.to_dict()["repaired"] is True


def test_valid_json_is_not_marked_repaired() -> None:
    reply = parse_verdict('{"verdict": "ACCEPT", "reason": "all six agree"}')

    assert reply.repaired is False
    assert reply.fields["reason"] == "all six agree"


def test_salvage_survives_a_brace_inside_the_reason() -> None:
    """Counts are quoted with braces, and the object also ends in one."""
    reply = parse_verdict(
        '{"verdict": "REJECT", "reason": "counts {2 Black, 1 Red} are "off"", '
        '"next_action": ""}'
    )

    assert reply.repaired is True
    assert reply.fields["reason"] == 'counts {2 Black, 1 Red} are "off"'
    # An empty JSON string stays an empty string; only an unquoted null is None.
    assert reply.fields["next_action"] == ""


def test_a_fenced_reply_still_parses() -> None:
    reply = parse_verdict('```json\n{"verdict": "RETRY"}\n```')

    assert reply.fields["verdict"] == "RETRY"


def test_salvage_refuses_a_reply_with_no_verdict_in_it() -> None:
    """Better no record than a confident-looking one with no decision."""
    with pytest.raises(VisionClientError, match="no verdict could be recovered"):
        parse_verdict('{"reason": "I am "unsure" about this one"}')


def test_a_reply_with_no_object_at_all_is_still_an_error() -> None:
    with pytest.raises(VisionClientError, match="No JSON object"):
        parse_verdict("I would rather not say.")


def test_an_unknown_verdict_word_is_refused() -> None:
    client = _client(_RecordingOpener(_reply('{"verdict": "MAYBE"}')), max_retries=0)

    with pytest.raises(VisionClientError, match="MAYBE"):
        client.judge(VisionRequest(sample_id="s", prompt="p"))


# -- the request that timed out ---------------------------------------------


def test_the_deadline_grows_with_the_images_attached() -> None:
    """The sample that timed out carried two crops and the whole frame."""
    opener = _RecordingOpener(_reply('{"verdict": "ACCEPT"}'))
    client = _client(opener)

    client.judge(
        VisionRequest(
            sample_id="s", prompt="p", crops=(b"a", b"b"), context_image=b"frame"
        )
    )

    assert opener.timeouts == [pytest.approx(120.0 + 45.0 * 3)]
    # The flat deadline this replaces is what expired on that sample.
    assert opener.timeouts[0] > VisionEndpointConfig().timeout_seconds


def test_a_request_with_no_images_keeps_the_base_deadline() -> None:
    opener = _RecordingOpener(_reply('{"verdict": "ACCEPT"}'))
    client = _client(opener)

    client.judge(VisionRequest(sample_id="s", prompt="p"))

    assert opener.timeouts == [pytest.approx(120.0)]


def test_the_deadline_is_configuration_not_a_constant() -> None:
    opener = _RecordingOpener(_reply('{"verdict": "ACCEPT"}'))
    client = _client(opener, timeout_seconds=10.0, timeout_per_image_seconds=1.0)

    client.judge(VisionRequest(sample_id="s", prompt="p", crops=(b"a",)))

    assert opener.timeouts == [pytest.approx(11.0)]


# -- the credential ---------------------------------------------------------


def test_the_key_never_reaches_an_error_message() -> None:
    error = urllib.error.HTTPError(
        "http://endpoint.invalid/v1/messages",
        401,
        "Unauthorized",
        {},  # type: ignore[arg-type]
        io.BytesIO(json.dumps({"error": f"bad key {KEY}"}).encode("utf-8")),
    )
    client = _client(_RaisingOpener(error), max_retries=0)

    with pytest.raises(VisionClientError) as caught:
        client.judge(VisionRequest(sample_id="s", prompt="p"))

    assert KEY not in str(caught.value)
    assert "<redacted>" in str(caught.value)


def test_the_endpoint_config_records_names_and_never_values() -> None:
    recorded = json.dumps(VisionEndpointConfig().to_dict())

    assert "VISION_LLM_API_KEY" in recorded
    assert KEY not in recorded


def test_refuses_to_call_an_endpoint_with_no_credential() -> None:
    with pytest.raises(VisionClientError, match="VISION_LLM_API_KEY"):
        HttpVisionLLMClient(
            VisionEndpointConfig(), environ={"VISION_LLM_URL": "http://x.invalid"}
        )


def test_refuses_an_endpoint_that_nobody_chose() -> None:
    with pytest.raises(VisionClientError, match="VISION_LLM_URL"):
        HttpVisionLLMClient(VisionEndpointConfig(), environ={"VISION_LLM_API_KEY": KEY})


# -- the guard against calls nobody asked for -------------------------------


def test_no_call_leaves_the_process_unless_calls_are_enabled() -> None:
    """The load-bearing protection: default deny, checked at the request."""
    opener = _RecordingOpener(_reply('{"verdict": "ACCEPT"}'))
    client = HttpVisionLLMClient(
        VisionEndpointConfig(),
        environ={"VISION_LLM_URL": "http://endpoint.invalid", "VISION_LLM_API_KEY": KEY},
        opener=opener,
    )

    with pytest.raises(VisionCallsDisabledError, match=CALLS_ENABLED_ENV):
        client.judge(VisionRequest(sample_id="s", prompt="p"))

    # Not merely refused -- nothing was sent.
    assert opener.timeouts == []


def test_the_guard_is_not_a_visionclienterror() -> None:
    """So the retry loop cannot swallow it and a report cannot record it.

    Retrying a permission refusal cannot help, and every remaining sample
    would fail the same way. It has to stop the run.
    """
    assert not issubclass(VisionCallsDisabledError, VisionClientError)


def test_a_truthy_value_is_not_enough_to_enable_calls() -> None:
    """An environment that sets flags to "1" wholesale must not enable this."""
    for value in ("1", "true", "yes", "ALLOW", ""):
        opener = _RecordingOpener(_reply('{"verdict": "ACCEPT"}'))
        client = HttpVisionLLMClient(
            VisionEndpointConfig(),
            environ={
                "VISION_LLM_URL": "http://endpoint.invalid",
                "VISION_LLM_API_KEY": KEY,
                CALLS_ENABLED_ENV: value,
            },
            opener=opener,
        )
        with pytest.raises(VisionCallsDisabledError):
            client.judge(VisionRequest(sample_id="s", prompt="p"))
        assert opener.timeouts == []


def test_permission_is_re_read_not_cached_at_construction() -> None:
    """A long-lived process must not keep a permission it was granted once."""
    environ = dict(ENVIRON)
    opener = _RecordingOpener(_reply('{"verdict": "ACCEPT"}'))
    client = HttpVisionLLMClient(
        VisionEndpointConfig(), environ=environ, opener=opener
    )
    client.judge(VisionRequest(sample_id="s", prompt="p"))
    assert len(opener.timeouts) == 1

    environ[CALLS_ENABLED_ENV] = "revoked"

    with pytest.raises(VisionCallsDisabledError):
        client.judge(VisionRequest(sample_id="s", prompt="p"))
    assert len(opener.timeouts) == 1


# -- talking to an endpoint that has no authentication ----------------------


def test_a_keyless_endpoint_needs_saying_so_not_a_placeholder() -> None:
    profile = openai_compatible_profile(
        model="some-local-model", url_env="LOCAL_URL", requires_credential=False
    )
    opener = _RecordingOpener(_reply('{"verdict": "ACCEPT"}'))

    client = HttpVisionLLMClient(
        profile,
        environ={"LOCAL_URL": "http://server.invalid/v1", CALLS_ENABLED_ENV: CALLS_ALLOWED},
        opener=opener,
    )
    client.judge(VisionRequest(sample_id="s", prompt="p"))

    assert opener.timeouts  # the call went out
    # No credential means no auth header at all, rather than an empty one.
    assert "authorization" not in client._headers()


def test_an_endpoint_with_no_credential_says_so_in_provenance() -> None:
    recorded = openai_compatible_profile(
        model="m", url_env="U", requires_credential=False
    ).to_dict()

    assert recorded["requires_credential"] is False


def test_the_profile_carries_no_address_of_its_own() -> None:
    """An endpoint baked in here is one copy away from a station using it."""
    recorded = openai_compatible_profile(model="m", url_env="SOME_URL").to_dict()

    assert recorded["url_env"] == "SOME_URL"
    assert "10." not in json.dumps(recorded)
    assert recorded["dialect"] == "chat_completions"


def test_a_credentialled_endpoint_still_refuses_without_a_key() -> None:
    with pytest.raises(VisionClientError, match="requires_credential=False"):
        HttpVisionLLMClient(
            openai_compatible_profile(
                model="m", url_env="U", key_env="K", requires_credential=True
            ),
            environ={"U": "http://server.invalid/v1"},
        )


# -- reporting across two different services --------------------------------


def test_openai_usage_is_counted_under_the_names_the_reports_total() -> None:
    usage = extract_usage(
        {"usage": {"prompt_tokens": 158, "completion_tokens": 153}}
    )

    assert usage["input_tokens"] == 158
    assert usage["output_tokens"] == 153
    # The original spelling survives; the record says what the service said.
    assert usage["prompt_tokens"] == 158


def test_anthropic_usage_is_left_exactly_as_it_came() -> None:
    usage = extract_usage({"usage": {"input_tokens": 10, "output_tokens": 5}})

    assert usage == {"input_tokens": 10, "output_tokens": 5}


def test_the_record_names_the_model_that_actually_answered() -> None:
    """Two reviewers averaged together is a corrupted dataset."""
    body = json.dumps(
        {
            "model": "Qwen3.8-27B-GGUF",
            "choices": [{"message": {"content": '{"verdict": "REJECT"}'}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    )
    client = _client(_RecordingOpener(body), model="asked-for-something-else")

    verdict = client.judge(VisionRequest(sample_id="s", prompt="p"))

    assert verdict.model == "Qwen3.8-27B-GGUF"
    assert verdict.to_dict()["model"] == "Qwen3.8-27B-GGUF"


def test_a_keyless_endpoint_is_sent_no_credential_from_the_environment() -> None:
    """A hosted provider's key must not follow the run to somebody's laptop.

    --key-env defaults to a hosted provider's variable, and that variable is
    routinely exported. Declaring an endpoint keyless has to mean no key
    leaves the process, not merely that none was asked for.
    """
    client = HttpVisionLLMClient(
        openai_compatible_profile(
            model="local", url_env="LOCAL_URL", key_env="VISION_LLM_API_KEY",
            requires_credential=False,
        ),
        environ={
            "LOCAL_URL": "http://server.invalid/v1",
            "VISION_LLM_API_KEY": KEY,  # exported, and must be ignored
            CALLS_ENABLED_ENV: CALLS_ALLOWED,
        },
        opener=_RecordingOpener(_reply('{"verdict": "ACCEPT"}')),
    )

    assert KEY not in json.dumps(client._headers())
    assert "authorization" not in client._headers()


def test_a_reply_truncated_before_the_answer_says_so() -> None:
    """A reasoning model out of budget returns thinking and no answer.

    Reported as truncation rather than "no JSON object", which sent the first
    real Qwen run chasing a parser bug that was not there.
    """
    body = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": "", "reasoning_content": "still thinking"},
                }
            ]
        }
    )
    client = _client(_RecordingOpener(body), max_retries=0)

    with pytest.raises(VisionClientError, match="max_tokens"):
        client.judge(VisionRequest(sample_id="s", prompt="p"))


def test_reasoning_is_never_mistaken_for_the_answer() -> None:
    """Parsing what the model was merely contemplating would invent verdicts."""
    body = json.dumps(
        {
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "content": "",
                        "reasoning_content": 'maybe {"verdict": "ACCEPT"}?',
                    },
                }
            ]
        }
    )
    client = _client(_RecordingOpener(body), max_retries=0)

    with pytest.raises(VisionClientError):
        client.judge(VisionRequest(sample_id="s", prompt="p"))


def test_the_local_profile_leaves_room_for_reasoning() -> None:
    assert openai_compatible_profile(model="m", url_env="U").max_tokens >= 2048
