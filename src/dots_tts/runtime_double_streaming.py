from __future__ import annotations

from pathlib import Path

import torch
from loguru import logger

from dots_tts.data.pipelines.tts_pipeline import TTS_INTERLEAVE_PREFIX
from dots_tts.runtime import DotsTtsRuntime
from dots_tts.utils.logging import categorized_log as logc
from dots_tts.utils.tokenizer import (
    AUDIO_GEN_END_TOKEN,
    AUDIO_GEN_SPAN_TOKEN,
    TEXT_COND_END_TOKEN,
    require_token_id,
)
from dots_tts.utils.util import get_dtype

PAIR_DOUBLE_STREAMING_TEMPLATE_NAME = "tts_interleave_pair"
DOUBLE_STREAMING_TEMPLATE_NAMES = frozenset(
    {"tts_interleave", PAIR_DOUBLE_STREAMING_TEMPLATE_NAME}
)
INTERLEAVE_MODE_ONE_TO_ONE = "one_to_one"
INTERLEAVE_MODE_BUFFERED_RATIO = "buffered_ratio"
INTERLEAVE_MODES = frozenset(
    {INTERLEAVE_MODE_ONE_TO_ONE, INTERLEAVE_MODE_BUFFERED_RATIO}
)
DEFAULT_INTERLEAVE_PATTERN = (INTERLEAVE_MODE_ONE_TO_ONE, 1, 0, 0)


def normalize_interleave_pattern(
    *,
    interleave_mode: str = INTERLEAVE_MODE_ONE_TO_ONE,
    initial_lookahead: int | None = None,
    ta_per_tta: int = 2,
    warmup_ta: int = 1,
) -> tuple[str, int, int, int]:
    if interleave_mode not in INTERLEAVE_MODES:
        raise ValueError(
            f"Unknown interleave_mode={interleave_mode!r}. "
            f"Expected one of {sorted(INTERLEAVE_MODES)}."
        )
    if interleave_mode == INTERLEAVE_MODE_ONE_TO_ONE:
        return interleave_mode, 1, 0, 0

    lookahead = 3 if initial_lookahead is None else int(initial_lookahead)
    ta_count = int(ta_per_tta)
    warmup_count = int(warmup_ta)
    if lookahead <= 0:
        raise ValueError("initial_lookahead must be positive.")
    if ta_count <= 0:
        raise ValueError("ta_per_tta must be positive for buffered_ratio.")
    if warmup_count < 0:
        raise ValueError("warmup_ta must be non-negative.")
    return interleave_mode, lookahead, ta_count, warmup_count


def _load_interleave_pattern_from_model_config(
    runtime: DotsTtsRuntime,
) -> tuple[str, int | None, int, int]:
    streaming = getattr(runtime.model.config, "streaming", None)
    if streaming is None:
        return DEFAULT_INTERLEAVE_PATTERN
    return (
        str(streaming.interleave_mode),
        streaming.initial_lookahead,
        int(streaming.ta_per_tta),
        int(streaming.warmup_ta),
    )


def interleave_text_quota(
    audio_index: int,
    *,
    interleave_mode: str = INTERLEAVE_MODE_ONE_TO_ONE,
    initial_lookahead: int | None = None,
    ta_per_tta: int = 2,
    warmup_ta: int = 1,
) -> int:
    mode, lookahead, ta_count, warmup_count = normalize_interleave_pattern(
        interleave_mode=interleave_mode,
        initial_lookahead=initial_lookahead,
        ta_per_tta=ta_per_tta,
        warmup_ta=warmup_ta,
    )
    if mode == INTERLEAVE_MODE_ONE_TO_ONE:
        return 1
    if int(audio_index) == 0:
        return lookahead
    steady_index = int(audio_index) - 1
    if steady_index < warmup_count:
        return 1
    return 2 if (steady_index - warmup_count) % (ta_count + 1) == ta_count else 1


def build_interleave_token_sequence(
    *,
    text_tokens: list[int],
    audio_tokens: list[int],
    text_cond_end_id: int,
    interleave_mode: str = INTERLEAVE_MODE_ONE_TO_ONE,
    initial_lookahead: int | None = None,
    ta_per_tta: int = 2,
    warmup_ta: int = 1,
) -> list[int]:
    token_ids: list[int] = []
    text_index = 0
    text_cond_end_added = False

    for audio_index, audio_token in enumerate(audio_tokens):
        quota = interleave_text_quota(
            audio_index,
            interleave_mode=interleave_mode,
            initial_lookahead=initial_lookahead,
            ta_per_tta=ta_per_tta,
            warmup_ta=warmup_ta,
        )
        had_text = text_index < len(text_tokens)
        for _ in range(quota):
            if text_index >= len(text_tokens):
                break
            token_ids.append(text_tokens[text_index])
            text_index += 1
        if not had_text and not text_cond_end_added:
            token_ids.append(text_cond_end_id)
            text_cond_end_added = True
        token_ids.append(audio_token)

    while text_index < len(text_tokens):
        token_ids.append(text_tokens[text_index])
        text_index += 1
    if not text_cond_end_added:
        token_ids.append(text_cond_end_id)
    return token_ids


class DoubleStreamingSession:
    """Incremental interleave session for text-token to audio-chunk generation."""

    def __init__(
        self,
        runtime: DotsTtsRuntime,
        *,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str = "tts_interleave",
        ode_method: str | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
        initial_silence_audio_tokens: int | None = None,
    ) -> None:
        normalized_prompt_text = runtime._process_prompt_text(prompt_text)
        if (
            template_name == "tts_interleave"
            and prompt_audio_path is not None
            and normalized_prompt_text
        ):
            template_name = PAIR_DOUBLE_STREAMING_TEMPLATE_NAME
        if template_name not in DOUBLE_STREAMING_TEMPLATE_NAMES:
            raise ValueError(
                f"Unknown double streaming template_name={template_name!r}. "
                f"Expected one of {sorted(DOUBLE_STREAMING_TEMPLATE_NAMES)}."
            )
        if template_name == PAIR_DOUBLE_STREAMING_TEMPLATE_NAME:
            if prompt_audio_path is None:
                raise ValueError("tts_interleave_pair requires prompt_audio_path.")
            if not normalized_prompt_text:
                raise ValueError("tts_interleave_pair requires prompt_text.")
        elif normalized_prompt_text:
            raise ValueError(
                "tts_interleave double streaming does not support prompt_text."
            )

        ode_method, num_steps, guidance_scale = runtime.resolve_sampling_options(
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
        )
        self.runtime = runtime
        self.model = runtime.model
        self.device = runtime.device
        self.template_name = template_name
        self.ode_method = ode_method
        self.num_steps = int(num_steps)
        self.guidance_scale = float(guidance_scale)
        self.speaker_scale = float(speaker_scale)
        self.eos_threshold = float(eos_threshold)
        self.max_generate_length = runtime.max_generate_length
        config_pattern = _load_interleave_pattern_from_model_config(runtime)
        (
            self.interleave_mode,
            self.initial_lookahead,
            self.ta_per_tta,
            self.warmup_ta,
        ) = normalize_interleave_pattern(
            interleave_mode=config_pattern[0],
            initial_lookahead=config_pattern[1],
            ta_per_tta=config_pattern[2],
            warmup_ta=config_pattern[3],
        )
        if initial_silence_audio_tokens is None:
            initial_silence_audio_tokens = (
                0 if template_name == PAIR_DOUBLE_STREAMING_TEMPLATE_NAME else 1
            )
        self._initial_silence_audio_tokens = max(
            0,
            min(10, int(initial_silence_audio_tokens or 0)),
        )

        self._dtype = get_dtype(runtime.precision)
        self._use_amp = self.device.type in {"cuda", "xpu"} and self._dtype in {
            torch.float16,
            torch.bfloat16,
        }
        self._prefix_token_ids = tuple(
            self.model.tokenizer.encode(
                TTS_INTERLEAVE_PREFIX,
                add_special_tokens=False,
            )
        )
        self._pending_token_ids = list(self._prefix_token_ids)
        self._state = self.model._allocate_generate_state(
            max_audio_patch_count=self.max_generate_length,
            device=self.device,
            dtype=self._dtype,
        )
        self._vocoder_inference = self.model._get_vocoder_inference()
        self._vocoder_state = self._vocoder_inference.init_stream_state(
            batch_size=1,
            chunk_size=self.model.core.latent_patch_size,
        )
        self._g_cond = None
        self._started = False
        self._text_finished = False
        self._closed = False
        self._decoded_patch_count = 0
        self._text_token_buffer: list[int] = []
        self._debug_audio_events: list[str] = []

        if template_name == PAIR_DOUBLE_STREAMING_TEMPLATE_NAME:
            self._prefill_pair_prompt(
                prompt_audio_path=prompt_audio_path,
                prompt_text=normalized_prompt_text,
            )
        elif prompt_audio_path is not None:
            self._prepare_ref_audio_only_conditioning(prompt_audio_path)

        logger.debug(
            logc(
                "stream",
                "Double streaming session started: template_name={} prefix_token_count={} precision={} "
                "ode_method={} num_steps={} guidance_scale={} speaker_scale={} max_audio_patch_count={} "
                "initial_silence_audio_tokens={} interleave_mode={} initial_lookahead={} "
                "ta_per_tta={} warmup_ta={} has_ref_audio_only={}",
            ),
            self.template_name,
            len(self._prefix_token_ids),
            runtime.precision,
            self.ode_method,
            self.num_steps,
            self.guidance_scale,
            self.speaker_scale,
            self.max_generate_length,
            self._initial_silence_audio_tokens,
            self.interleave_mode,
            self.initial_lookahead,
            self.ta_per_tta,
            self.warmup_ta,
            self._g_cond is not None
            and self.template_name != PAIR_DOUBLE_STREAMING_TEMPLATE_NAME,
        )

    @property
    def is_finished(self) -> bool:
        return self._closed

    def push_text_token(self, text_token: int) -> torch.Tensor | None:
        self._ensure_active()
        if self._text_finished:
            raise RuntimeError("Cannot push text tokens after finish_text().")
        if self._state.end_flag:
            raise RuntimeError(
                "Double streaming generation has already reached EOS. "
                "Call finish_text() to flush the remaining audio tail."
            )

        token_id = int(text_token)
        self._text_token_buffer.append(token_id)
        quota = interleave_text_quota(
            self._decoded_patch_count,
            interleave_mode=self.interleave_mode,
            initial_lookahead=self.initial_lookahead,
            ta_per_tta=self.ta_per_tta,
            warmup_ta=self.warmup_ta,
        )
        if len(self._text_token_buffer) < quota:
            self._debug_audio_events.append("wait")
            return None

        chunk_token_ids = self._pop_text_chunk(quota)
        self._consume_text_chunk(chunk_token_ids)
        self._debug_audio_events.append("audio")
        return self._decode_audio_chunk()

    def finish_text(self):
        self._ensure_active()

        if not self._state.end_flag:
            if not self._text_finished:
                text_end_chunk = [
                    *self._text_token_buffer,
                    self.model.core.text_cond_end_id,
                ]
                self._text_token_buffer.clear()
                if not self._started:
                    text_end_chunk = [*self._pending_token_ids, *text_end_chunk]
                    self._pending_token_ids.clear()
                    self._started = True
                self._consume_text_chunk(text_end_chunk)
                self._text_finished = True

            while not self._state.end_flag:
                audio_chunk = self._decode_audio_chunk(continue_audio_span=True)
                if audio_chunk is not None:
                    yield audio_chunk
        else:
            self._text_finished = True

        final_chunk = self._vocoder_inference.flush(self._vocoder_state)
        self._closed = True
        logger.info(
            logc("stream", "Double streaming session finished: decoded_patch_count={}"),
            self._decoded_patch_count,
        )
        if final_chunk.size(-1) > 0:
            yield final_chunk

    def _ensure_active(self) -> None:
        if self._closed:
            raise RuntimeError("Double streaming session is already closed.")

    def _pop_text_chunk(self, quota: int) -> list[int]:
        chunk_token_ids = self._text_token_buffer[:quota]
        del self._text_token_buffer[:quota]
        if not self._started:
            chunk_token_ids = [*self._pending_token_ids, *chunk_token_ids]
            self._pending_token_ids.clear()
            self._started = True
        return chunk_token_ids

    def _prepare_ref_audio_only_conditioning(self, prompt_audio_path: str) -> None:
        cache = getattr(self.runtime, "_double_streaming_prompt_g_cond_cache", None)
        if cache is None:
            cache = {}
            setattr(self.runtime, "_double_streaming_prompt_g_cond_cache", cache)
        prompt_cache_key = (
            str(Path(prompt_audio_path).expanduser().resolve()),
            str(self.device),
            str(self._dtype),
            self.speaker_scale,
        )
        cached_g_cond = cache.get(prompt_cache_key)
        if cached_g_cond is None:
            prompt_audio = self.runtime._load_prompt_audio(prompt_audio_path)
            with torch.no_grad():
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self._dtype,
                    enabled=self._use_amp,
                ):
                    prompt_conditioning = self.model._prepare_prompt_conditioning(
                        prompt_audio,
                        use_prompt_prefill=False,
                        speaker_scale=self.speaker_scale,
                    )
            cached_g_cond = prompt_conditioning.g_cond.detach()
            cache[prompt_cache_key] = cached_g_cond
            logger.debug(
                logc(
                    "cache",
                    "Double streaming prompt conditioning cached: path={} device={} "
                    "dtype={} speaker_scale={}",
                ),
                prompt_cache_key[0],
                self.device,
                self._dtype,
                self.speaker_scale,
            )
        else:
            logger.debug(
                logc(
                    "cache",
                    "Double streaming prompt conditioning cache hit: path={} device={} "
                    "dtype={} speaker_scale={}",
                ),
                prompt_cache_key[0],
                self.device,
                self._dtype,
                self.speaker_scale,
            )
        self._g_cond = cached_g_cond

    def _prefill_pair_prompt(
        self,
        *,
        prompt_audio_path: str | None,
        prompt_text: str,
    ) -> None:
        if prompt_audio_path is None:
            raise ValueError("tts_interleave_pair requires prompt_audio_path.")
        prompt_audio = self.runtime._load_prompt_audio(prompt_audio_path)
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=self._use_amp,
            ):
                prompt_conditioning = self.model._prepare_prompt_conditioning(
                    prompt_audio,
                    use_prompt_prefill=True,
                    speaker_scale=self.speaker_scale,
                )
                if (
                    prompt_conditioning.prompt_patches is None
                    or prompt_conditioning.prompt_latents is None
                ):
                    raise RuntimeError(
                        "tts_interleave_pair prompt prefill did not produce prompt latents."
                    )
                prompt_patch_count = int(prompt_conditioning.prompt_patches.size(1))
                prompt_audio_span_count = prompt_patch_count + 1
                if self.max_generate_length <= prompt_audio_span_count:
                    raise ValueError(
                        "max_generate_length must exceed prompt audio span count for "
                        "tts_interleave_pair double streaming: "
                        f"max_generate_length={self.max_generate_length} "
                        f"prompt_audio_span_count={prompt_audio_span_count}."
                    )
                prompt_schedule = self._build_pair_prompt_schedule(
                    prompt_text=prompt_text,
                    prompt_audio_span_count=prompt_audio_span_count,
                )
                audio_placeholder_ids = set(self.model.core.audio_span_token_ids)
                span_positions = self.model._find_audio_span_positions(
                    prompt_schedule,
                    audio_placeholder_ids=audio_placeholder_ids,
                )
                prompt_patch_embeddings = self.model._prefill_prompt_latents(
                    prompt_conditioning.prompt_latents,
                    state=self._state,
                )
                tail_position = self.model._prefill(
                    prompt_schedule,
                    state=self._state,
                    span_positions=span_positions,
                    prompt_patches=prompt_conditioning.prompt_patches,
                    prompt_patch_embeddings=prompt_patch_embeddings,
                    audio_placeholder_ids=audio_placeholder_ids,
                )
                prompt_tail_patch = self.model._decode_next_audio(
                    self._state,
                    g_cond=prompt_conditioning.g_cond,
                    ode_method=self.ode_method,
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                )
                self.model._consume_audio_patch(
                    self._state,
                    audio_patch=prompt_tail_patch,
                )

        self._g_cond = (
            None
            if prompt_conditioning.g_cond is None
            else prompt_conditioning.g_cond.detach()
        )
        self._pending_token_ids = [
            *prompt_schedule[0, tail_position + 1 :].tolist(),
            *self._prefix_token_ids,
        ]
        self.max_generate_length -= prompt_audio_span_count
        logger.debug(
            logc(
                "conditioning",
                "Double streaming pair prompt prefetched: prompt_audio_span_count={} "
                "target_max_audio_patch_count={} pending_token_count={}",
            ),
            prompt_audio_span_count,
            self.max_generate_length,
            len(self._pending_token_ids),
        )

    def _build_pair_prompt_schedule(
        self,
        *,
        prompt_text: str,
        prompt_audio_span_count: int,
    ) -> torch.Tensor:
        tokenizer = self.model.tokenizer
        audio_span_id = require_token_id(tokenizer, AUDIO_GEN_SPAN_TOKEN)
        audio_end_id = require_token_id(tokenizer, AUDIO_GEN_END_TOKEN)
        text_cond_end_id = require_token_id(tokenizer, TEXT_COND_END_TOKEN)
        prompt_text_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        audio_tokens = [audio_span_id] * int(prompt_audio_span_count) + [audio_end_id]
        token_ids = list(self._prefix_token_ids)
        token_ids.extend(
            build_interleave_token_sequence(
                text_tokens=prompt_text_tokens,
                audio_tokens=audio_tokens,
                text_cond_end_id=text_cond_end_id,
                interleave_mode=self.interleave_mode,
                initial_lookahead=self.initial_lookahead,
                ta_per_tta=self.ta_per_tta,
                warmup_ta=self.warmup_ta,
            )
        )
        return torch.tensor([token_ids], dtype=torch.long, device=self.device)

    def _consume_text_chunk(self, token_ids: list[int]) -> None:
        schedule = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )
        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=self._use_amp,
            ):
                self.model._consume_text_schedule(
                    schedule,
                    position=0,
                    next_audio_position=schedule.size(1),
                    state=self._state,
                )

    def _get_initial_silence_audio_patch(
        self,
        patch_index: int,
        audio_patch: torch.Tensor,
    ) -> torch.Tensor:
        cache = getattr(
            self.runtime, "_double_streaming_silence_audio_patch_cache", None
        )
        if cache is None:
            cache = {}
            setattr(self.runtime, "_double_streaming_silence_audio_patch_cache", cache)

        cache_count = 10
        patch_size = int(self.model.core.latent_patch_size)
        key = (
            str(self.device),
            str(self._dtype),
            patch_size,
            int(audio_patch.size(-1)),
            cache_count,
        )
        cached_patches = cache.get(key)
        if cached_patches is None:
            hop_size = int(getattr(self.model.vocoder, "hop_size", 1))
            zero_samples = cache_count * patch_size * hop_size
            zero_audio = torch.zeros(
                (1, 1, zero_samples),
                device=self.device,
                dtype=torch.float32,
            )
            silence_latents = self._vocoder_inference.extract_latents(zero_audio)
            silence_latents, _ = torch.split(
                silence_latents,
                int(audio_patch.size(-1)),
                dim=1,
            )
            silence_latents = silence_latents.transpose(1, 2)
            target_frames = cache_count * patch_size
            if silence_latents.size(1) < target_frames:
                silence_latents = torch.cat(
                    [
                        silence_latents,
                        silence_latents.new_zeros(
                            (
                                silence_latents.size(0),
                                target_frames - silence_latents.size(1),
                                silence_latents.size(2),
                            )
                        ),
                    ],
                    dim=1,
                )
            silence_latents = silence_latents[:, :target_frames, :]
            cached_patches = self.model.core.io_helper.normalize(silence_latents)
            cached_patches = cached_patches.to(
                device=self.device, dtype=audio_patch.dtype
            )
            cached_patches = cached_patches.reshape(
                1,
                cache_count,
                patch_size,
                int(audio_patch.size(-1)),
            ).detach()
            cache[key] = cached_patches
            logger.debug(
                logc(
                    "cache",
                    "Double streaming initial silence cache built: patches={} patch_size={} "
                    "hop_size={} device={} dtype={}",
                ),
                cache_count,
                patch_size,
                hop_size,
                self.device,
                audio_patch.dtype,
            )
        return cached_patches[:, int(patch_index)].clone()

    def _consume_audio_patch(self, audio_patch: torch.Tensor) -> None:
        self.model._consume_audio_patch(self._state, audio_patch=audio_patch)

    def _decode_audio_chunk(
        self, *, continue_audio_span: bool = False
    ) -> torch.Tensor | None:
        if self._decoded_patch_count >= self.max_generate_length:
            raise RuntimeError(
                "Double streaming exceeded max_generate_length before reaching EOS."
            )

        with torch.no_grad():
            with torch.autocast(
                device_type=self.device.type,
                dtype=self._dtype,
                enabled=self._use_amp,
            ):
                stop_after_current_audio = (
                    self.model._should_stop_after_current_audio(
                        self._state,
                        eos_threshold=self.eos_threshold,
                    )
                    if self._text_finished
                    else False
                )
                audio_patch = self.model._decode_next_audio(
                    self._state,
                    g_cond=self._g_cond,
                    ode_method=self.ode_method,
                    num_steps=self.num_steps,
                    guidance_scale=self.guidance_scale,
                )
                if self._decoded_patch_count < self._initial_silence_audio_tokens:
                    audio_patch = self._get_initial_silence_audio_patch(
                        self._decoded_patch_count,
                        audio_patch,
                    )
                self._consume_audio_patch(audio_patch)
                if continue_audio_span:
                    self.model._append_hidden_chunk(
                        self._state, self._state.llm_hiddens
                    )
                self._decoded_patch_count += 1
                latent_patch = self.model.core.io_helper.denormalize(audio_patch)
                audio_chunk = self._vocoder_inference.stream_step(
                    latent_patch,
                    self._vocoder_state,
                    optimize=self.model._optimize_enabled,
                    profile_step=self._decoded_patch_count,
                )
                if stop_after_current_audio:
                    self._state.end_flag = True

        if audio_chunk.size(-1) == 0:
            return None
        return audio_chunk


class DotsTtsRuntimeDoubleStreaming(DotsTtsRuntime):
    def start_double_streaming(
        self,
        *,
        prompt_audio_path: str | None = None,
        prompt_text: str | None = None,
        template_name: str = "tts_interleave",
        ode_method: str | None = None,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
        initial_silence_audio_tokens: int | None = None,
    ) -> DoubleStreamingSession:
        return DoubleStreamingSession(
            self,
            prompt_audio_path=prompt_audio_path,
            prompt_text=prompt_text,
            template_name=template_name,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            speaker_scale=speaker_scale,
            eos_threshold=eos_threshold,
            initial_silence_audio_tokens=initial_silence_audio_tokens,
        )


__all__ = ["DotsTtsRuntimeDoubleStreaming", "DoubleStreamingSession"]
