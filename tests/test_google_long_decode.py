"""
Tests for Gemini 4 (argon) in the Google LLM plugin: thinking is configured with a level
rather than a token budget, sampling parameters are dropped, the output ceiling is pinned
because the surfaces disagree on a default, and a turn too large for one request is resumed
through continuation tokens.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import pytest
from google.genai import types

from livekit.agents import llm as agents_llm
from livekit.plugins.google.llm import (
    LLM,
    LONG_DECODE_CONTEXT_WINDOW,
    LONG_DECODE_HTTP_TIMEOUT_MS,
    LONG_DECODE_MAX_HOPS,
    LONG_DECODE_MIN_OUTPUT_TOKENS,
    _ignores_sampling_params,
    _is_gemini_4_model,
    _requires_thought_signatures,
    _supports_thinking_level,
)

pytestmark = pytest.mark.unit

ARGON = "gemini-4-argon"


def _llm(model: str = ARGON, **kwargs: Any) -> LLM:
    return LLM(model=model, api_key="test-key", **kwargs)


def _response(
    *,
    text: str | None = None,
    finish_reason: types.FinishReason | None = None,
    continuation_token: bytes | None = None,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> types.GenerateContentResponse:
    candidate = types.Candidate(
        content=types.Content(parts=[types.Part(text=text)]) if text is not None else None,
        finish_reason=finish_reason,
        continuation_token=continuation_token,
    )
    return types.GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=(
            types.GenerateContentResponseUsageMetadata(
                prompt_token_count=prompt_tokens,
                candidates_token_count=completion_tokens,
                total_token_count=(prompt_tokens or 0) + (completion_tokens or 0),
            )
            if prompt_tokens is not None or completion_tokens is not None
            else None
        ),
    )


class _FakeStreams:
    """Serves one scripted stream per call and records the config it was given."""

    def __init__(self, hops: list[list[types.GenerateContentResponse]]) -> None:
        self._hops = hops
        self.configs: list[types.GenerateContentConfig] = []

    async def __call__(self, **kwargs: Any) -> AsyncIterator[types.GenerateContentResponse]:
        self.configs.append(kwargs["config"])
        responses = self._hops[min(len(self.configs) - 1, len(self._hops) - 1)]

        async def _iter() -> AsyncIterator[types.GenerateContentResponse]:
            for response in responses:
                yield response

        return _iter()


async def _request_config(model: LLM, **chat_kwargs: Any) -> dict[str, Any]:
    """The config `chat()` built, with the stream closed again."""
    async with model.chat(chat_ctx=agents_llm.ChatContext.empty(), **chat_kwargs) as stream:
        return dict(stream._extra_kwargs)


async def _collect(model: LLM, streams: _FakeStreams) -> list[str]:
    text, _ = await _collect_with_usage(model, streams)
    return text


async def _collect_with_usage(
    model: LLM, streams: _FakeStreams
) -> tuple[list[str], agents_llm.CompletionUsage | None]:
    """The text chunks, and the usage a reader keeping only the last one would end up with."""
    model._client.aio.models.generate_content_stream = streams  # type: ignore[method-assign]
    chunks: list[str] = []
    usage: agents_llm.CompletionUsage | None = None
    async with model.chat(chat_ctx=agents_llm.ChatContext.empty()) as stream:
        async for chunk in stream:
            if chunk.delta and chunk.delta.content:
                chunks.append(chunk.delta.content)
            if chunk.usage is not None:
                usage = chunk.usage
    return chunks, usage


class TestModelDetection:
    @pytest.mark.parametrize(
        "model,expected",
        [
            ("gemini-4-argon", True),
            # the EAP name is not a listed id, but routes the same if passed as a raw string
            ("gemini-4-argon-eap", True),
            ("gemini-3-pro-preview", False),
            ("gemini-2.5-flash", False),
            ("gemma-4-27b-it", False),
        ],
    )
    def test_is_gemini_4(self, model: str, expected: bool) -> None:
        assert _is_gemini_4_model(model) is expected

    async def test_gemini_4_joins_the_gates_gemini_3_already_passed(self) -> None:
        # thinking_budget returns 400 on argon, so it must take the level path
        assert _supports_thinking_level(ARGON)
        assert _requires_thought_signatures(ARGON)

    async def test_only_gemini_4_drops_sampling_params(self) -> None:
        assert _ignores_sampling_params(ARGON)
        assert not _ignores_sampling_params("gemini-3-pro-preview")


class TestRequestShaping:
    async def test_sampling_params_are_dropped(self) -> None:
        extra = await _request_config(_llm(temperature=0.8, top_p=0.9, top_k=40))
        assert "temperature" not in extra
        assert "top_p" not in extra
        assert "top_k" not in extra

    async def test_sampling_params_survive_on_older_models(self) -> None:
        extra = await _request_config(_llm("gemini-3-pro-preview", temperature=0.8))
        assert extra["temperature"] == 0.8

    async def test_the_output_ceiling_is_pinned_when_unset(self) -> None:
        extra = await _request_config(_llm())
        assert extra["max_output_tokens"] == LONG_DECODE_MIN_OUTPUT_TOKENS

    async def test_an_explicit_ceiling_is_left_alone(self) -> None:
        # a voice agent wanting short replies keeps them; the risk is logged, not overridden
        extra = await _request_config(_llm(max_output_tokens=256))
        assert extra["max_output_tokens"] == 256

    async def test_older_models_keep_their_own_default(self) -> None:
        extra = await _request_config(_llm("gemini-2.5-flash"))
        assert "max_output_tokens" not in extra

    async def test_a_thinking_budget_is_replaced_by_a_level(self) -> None:
        extra = await _request_config(
            _llm(thinking_config=types.ThinkingConfig(thinking_budget=1024, thinking_level="high"))
        )
        cfg = extra["thinking_config"]
        assert cfg.thinking_budget is None
        assert cfg.thinking_level == types.ThinkingLevel.HIGH


class TestLongDecode:
    async def test_a_continuation_is_resumed_with_its_token(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="part one "),
                    _response(
                        finish_reason=types.FinishReason.CONTINUATION, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text="part two", finish_reason=types.FinishReason.STOP)],
            ]
        )
        assert await _collect(_llm(), streams) == ["part one ", "part two"]
        assert len(streams.configs) == 2
        assert streams.configs[0].continuation_token is None
        assert streams.configs[1].continuation_token == b"tok-1"
        # the budget is unchanged: the run did not run out of room, it was split
        assert streams.configs[1].max_output_tokens == LONG_DECODE_MIN_OUTPUT_TOKENS

    async def test_running_out_of_room_resumes_with_a_larger_budget(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="cut off", prompt_tokens=1000),
                    _response(
                        finish_reason=types.FinishReason.MAX_TOKENS, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text=" and the rest", finish_reason=types.FinishReason.STOP)],
            ]
        )
        assert await _collect(_llm(), streams) == ["cut off", " and the rest"]
        assert streams.configs[1].continuation_token == b"tok-1"
        assert streams.configs[1].max_output_tokens == LONG_DECODE_MIN_OUTPUT_TOKENS * 2

    async def test_the_budget_cannot_outgrow_what_the_prompt_leaves(self) -> None:
        prompt_tokens = LONG_DECODE_CONTEXT_WINDOW - LONG_DECODE_MIN_OUTPUT_TOKENS
        streams = _FakeStreams(
            [
                [
                    _response(text="cut off", prompt_tokens=prompt_tokens),
                    _response(
                        finish_reason=types.FinishReason.MAX_TOKENS, continuation_token=b"tok-1"
                    ),
                ],
            ]
        )
        assert await _collect(_llm(), streams) == ["cut off"]
        # no headroom to grow into, so the answer comes back truncated rather than looping
        assert len(streams.configs) == 1

    async def test_a_token_alongside_another_reason_does_not_resume(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="done"),
                    _response(finish_reason=types.FinishReason.STOP, continuation_token=b"tok-1"),
                ],
            ]
        )
        assert await _collect(_llm(), streams) == ["done"]
        assert len(streams.configs) == 1

    async def test_hops_are_capped(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="x"),
                    _response(
                        finish_reason=types.FinishReason.CONTINUATION, continuation_token=b"tok"
                    ),
                ],
            ]
        )
        await _collect(_llm(), streams)
        assert len(streams.configs) == LONG_DECODE_MAX_HOPS

    async def test_older_models_never_continue(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="done"),
                    _response(
                        finish_reason=types.FinishReason.CONTINUATION, continuation_token=b"tok"
                    ),
                ],
            ]
        )
        assert await _collect(_llm("gemini-2.5-flash"), streams) == ["done"]
        assert len(streams.configs) == 1

    async def test_the_read_timeout_covers_a_long_reasoning_turn(self) -> None:
        streams = _FakeStreams([[_response(text="hi", finish_reason=types.FinishReason.STOP)]])
        await _collect(_llm(), streams)
        http_options = streams.configs[0].http_options
        assert http_options is not None
        assert http_options.timeout == LONG_DECODE_HTTP_TIMEOUT_MS


class TestCallerBudget:
    async def test_an_explicit_ceiling_is_not_raised_to_continue(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="cut off", prompt_tokens=100),
                    _response(
                        finish_reason=types.FinishReason.MAX_TOKENS, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text=" and more", finish_reason=types.FinishReason.STOP)],
            ]
        )
        # the caller asked for a short reply; spending their budget ends the turn
        assert await _collect(_llm(max_output_tokens=256), streams) == ["cut off"]
        assert [c.max_output_tokens for c in streams.configs] == [256]

    async def test_a_ceiling_this_plugin_picked_is_still_raised(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="cut off", prompt_tokens=100),
                    _response(
                        finish_reason=types.FinishReason.MAX_TOKENS, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text=" and more", finish_reason=types.FinishReason.STOP)],
            ]
        )
        assert await _collect(_llm(), streams) == ["cut off", " and more"]
        assert [c.max_output_tokens for c in streams.configs] == [
            LONG_DECODE_MIN_OUTPUT_TOKENS,
            LONG_DECODE_MIN_OUTPUT_TOKENS * 2,
        ]

    async def test_an_explicit_ceiling_still_follows_a_plain_continuation(self) -> None:
        # CONTINUATION is the model splitting delivery, not asking for more budget
        streams = _FakeStreams(
            [
                [
                    _response(text="part one"),
                    _response(
                        finish_reason=types.FinishReason.CONTINUATION, continuation_token=b"tok-1"
                    ),
                ],
                [_response(text=" part two", finish_reason=types.FinishReason.STOP)],
            ]
        )
        assert await _collect(_llm(max_output_tokens=256), streams) == ["part one", " part two"]
        assert [c.max_output_tokens for c in streams.configs] == [256, 256]


class TestUsageAcrossHops:
    async def test_the_last_chunk_covers_the_whole_turn(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(text="one", prompt_tokens=100, completion_tokens=1000),
                    _response(
                        finish_reason=types.FinishReason.CONTINUATION, continuation_token=b"tok-1"
                    ),
                ],
                [
                    _response(
                        text=" two",
                        finish_reason=types.FinishReason.STOP,
                        prompt_tokens=100,
                        completion_tokens=1000,
                    )
                ],
            ]
        )
        _, usage = await _collect_with_usage(_llm(), streams)
        assert usage is not None
        # a reader keeping only the last chunk would otherwise see one hop's worth
        assert usage.completion_tokens == 2000
        assert usage.prompt_tokens == 200
        assert usage.total_tokens == 2200

    async def test_a_single_hop_reports_what_the_api_reported(self) -> None:
        streams = _FakeStreams(
            [
                [
                    _response(
                        text="hi",
                        finish_reason=types.FinishReason.STOP,
                        prompt_tokens=100,
                        completion_tokens=50,
                    )
                ]
            ]
        )
        _, usage = await _collect_with_usage(_llm(), streams)
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (100, 50)

    async def test_restated_counters_within_a_hop_are_not_added_up(self) -> None:
        # Gemini restates the request's totals on each chunk, so the last one wins per hop
        streams = _FakeStreams(
            [
                [
                    _response(text="a", prompt_tokens=100, completion_tokens=10),
                    _response(text="b", prompt_tokens=100, completion_tokens=20),
                    _response(
                        finish_reason=types.FinishReason.STOP,
                        prompt_tokens=100,
                        completion_tokens=30,
                    ),
                ]
            ]
        )
        _, usage = await _collect_with_usage(_llm(), streams)
        assert usage is not None
        assert (usage.prompt_tokens, usage.completion_tokens) == (100, 30)
