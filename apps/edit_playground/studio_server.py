"""Session API and Gradio queue endpoints for the custom studio frontend."""

from __future__ import annotations

import hashlib
import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import quote

import gradio as gr
import numpy as np
import soundfile as sf
import torch
from fastapi import Body, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from scipy.signal import resample_poly

from apps.edit_playground.constants import (
    DEFAULT_EDIT_SOURCE_PRESET_NAME,
    DEFAULT_PROMPT_PRESET_NAME,
    DEFAULT_TARGET_TEXT,
    EDIT_SOURCE_PRESET_NAMES,
    SUPPORTED_ODE_METHODS,
    VOICE_PROMPT_PRESET_NAMES,
)
from apps.edit_playground.editing import CompiledEdit, EditOperation, compile_edit
from apps.edit_playground.history import GenerationHistoryStore, history_allowed
from apps.edit_playground.preset_operations import (
    EDIT_SOURCE_PRESET_BY_NAME,
    EDIT_SOURCE_PRESET_SPECS,
    build_segmented_preset_operations,
)
from apps.edit_playground.recognition import RecognitionJobStore
from apps.edit_playground.service import (
    EditRequest,
    GenerationResult,
    GenerationSettings,
    StudioService,
    TTSRequest,
    _safe_session_id,
)
from apps.edit_playground.state import (
    AudioReference,
    AudioSegment,
    NoiseOverlay,
    NoiseUpload,
    RevisionNode,
    SessionState,
    UploadedAudio,
    remove_pruned_audio,
)
from dots_tts.data.edit_instruction import (
    EditXVectorMode,
    concat_edit_instruction_values,
    normalize_edit_xvector_mode,
    validate_literal_text,
)
from dots_tts.data.noise_mixing import (
    fit_noise_segment,
    limit_peak,
    mix_region_at_snr,
)
from dots_tts.utils.audio import high_quality_resample

DESTINATIONS = frozenset({"prompt", "edit_source"})
MAX_SOURCE_SEGMENTS = 3
MAX_SOURCE_SECONDS = 30.0
MAX_AUDIO_SECONDS = 30.0
MAX_UPLOAD_BYTES = 8_192 * 1_024
COMPOSITE_SAMPLE_RATE = 48_000


class StartupState:
    """Small thread-safe readiness state exposed before expensive startup work."""

    def __init__(
        self,
        *,
        frontend: str,
        model: str,
        warmup: str | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._values: dict[str, Any] = {
            "frontend": frontend,
            "model": model,
            "frontend_error": None,
            "model_error": None,
            "warmup": warmup or ("running" if model == "warming" else "complete"),
        }

    def set(self, component: str, status: str, error: str | None = None) -> None:
        with self._lock:
            self._values[component] = status
            self._values[f"{component}_error"] = error

    def set_warmup(self, status: str) -> None:
        with self._lock:
            self._values["warmup"] = status

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            values = dict(self._values)
        values["ready"] = values["frontend"] == "ready" and values["model"] == "ready"
        return values


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _settings(payload: Mapping[str, Any] | None) -> GenerationSettings:
    value = payload or {}
    return GenerationSettings(
        ode_method=str(value.get("ode_method", "euler")),
        num_steps=int(value.get("num_steps", 32)),
        guidance_scale=float(value.get("guidance_scale", 1.0)),
        speaker_scale=float(value.get("speaker_scale", 1.5)),
        seed=int(value.get("seed", 20260414)),
    ).normalized()


def _use_xvector(
    payload: Mapping[str, Any],
    *,
    allowed: bool,
    default: bool,
) -> bool:
    value = payload.get("use_xvector", default)
    if not isinstance(value, bool):
        raise ValueError("use_xvector must be a boolean.")
    if value and not allowed:
        raise ValueError("Speaker guidance is unavailable for this audio.")
    return value


def _edit_xvector_mode(payload: Mapping[str, Any]) -> EditXVectorMode:
    return normalize_edit_xvector_mode(payload.get("use_xvector", "auto"))


def _optional_bool(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: bool = False,
) -> bool:
    value = payload.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{key} must be a boolean.")
    return value


class StudioSessionStore:
    """Thread-safe browser-session state with immutable revision nodes."""

    def __init__(
        self,
        service: StudioService,
        *,
        history_store: GenerationHistoryStore | None = None,
        uploads_enabled: bool = True,
    ) -> None:
        self.service = service
        self.history_store = history_store
        self.uploads_enabled = uploads_enabled
        self._states: dict[str, SessionState] = {}
        self._locks: dict[str, threading.RLock] = {}
        self._guard = threading.RLock()
        self._presets = {preset.name: preset for preset in service.prompt_presets}
        self._prompt_presets = tuple(
            preset
            for preset in service.prompt_presets
            if preset.name in VOICE_PROMPT_PRESET_NAMES
        )
        self._edit_source_preset_specs = tuple(
            spec
            for spec in EDIT_SOURCE_PRESET_SPECS
            if spec.name in EDIT_SOURCE_PRESET_NAMES
            and all(name in self._presets for name in spec.segment_presets)
        )
        self._noise_presets: dict[str, NoiseUpload] = {}
        for path in sorted(service.config.noise_presets_dir.glob("*.wav")):
            try:
                duration = self._duration(str(path))
            except ValueError:
                continue
            preset_id = f"noise-{hashlib.sha1(path.name.encode(), usedforsecurity=False).hexdigest()[:16]}"
            self._noise_presets[preset_id] = NoiseUpload(
                id=preset_id,
                name=path.stem,
                audio_path=str(path.resolve()),
                duration_seconds=duration,
                created_at="",
            )
        self._default_prompt_preset = next(
            (
                item
                for item in self._prompt_presets
                if item.name == DEFAULT_PROMPT_PRESET_NAME
            ),
            self._prompt_presets[0] if self._prompt_presets else None,
        )
        self._default_edit_source_preset = next(
            (
                item
                for item in self._edit_source_preset_specs
                if item.name == DEFAULT_EDIT_SOURCE_PRESET_NAME
            ),
            self._edit_source_preset_specs[0]
            if self._edit_source_preset_specs
            else None,
        )

    def _require_uploads_enabled(self) -> None:
        if not self.uploads_enabled:
            raise ValueError("Uploads are disabled for this deployment.")

    @staticmethod
    def _duration(path: str) -> float:
        try:
            return float(sf.info(path).duration)
        except Exception as exc:
            raise ValueError("Audio could not be decoded.") from exc

    def _reference(
        self,
        audio_path: str,
        transcript: str,
        *,
        revision_id: str | None = None,
        multiple_speakers: bool = False,
        allow_xvector: bool = True,
        segment_id: str | None = None,
        history_consent: bool = True,
    ) -> AudioReference:
        text = transcript.strip()
        if not text:
            raise ValueError("Transcript must not be empty.")
        segment = AudioSegment(
            id=segment_id or uuid.uuid4().hex,
            audio_path=audio_path,
            original_transcript=text,
            duration_seconds=self._duration(audio_path),
            revision_id=revision_id,
            multiple_speakers=multiple_speakers,
            allow_xvector=allow_xvector and not multiple_speakers,
            history_consent=history_consent,
        )
        return AudioReference(
            audio_path=audio_path,
            text=text,
            revision_id=revision_id,
            segments=(segment,),
        )

    @staticmethod
    def _is_cjk(char: str) -> bool:
        if not char:
            return False
        value = ord(char)
        return (
            0x3400 <= value <= 0x4DBF
            or 0x4E00 <= value <= 0x9FFF
            or 0xF900 <= value <= 0xFAFF
            or 0x20000 <= value <= 0x3134F
        )

    @classmethod
    def _join_transcripts(cls, parts: list[str]) -> str:
        result = ""
        for raw in parts:
            part = raw.strip()
            if not part:
                continue
            if not result:
                result = part
            else:
                separator = (
                    "" if cls._is_cjk(result[-1]) and cls._is_cjk(part[0]) else " "
                )
                result += separator + part
        return result

    @classmethod
    def _separator(cls, left: str, right: str) -> str:
        left = left.rstrip()
        right = right.lstrip()
        if not left or not right:
            return ""
        return "" if cls._is_cjk(left[-1]) and cls._is_cjk(right[0]) else " "

    def compile_segment_edit(
        self,
        session_id: str,
        operations: list[Mapping[str, Any]],
        expected_version: str,
        *,
        enhance: bool = False,
    ) -> tuple[str, CompiledEdit]:
        """Compile segment-local operations in the current audio order."""

        with self.lock(session_id):
            source = self.get(session_id).edit_source
            if source is None:
                raise ValueError("Choose an Edit Source audio first.")
            self._check_version(source, expected_version)
            by_segment: dict[str, list[EditOperation]] = {
                segment.id: [] for segment in source.segments
            }
            seen_ids: set[str] = set()
            for value in operations:
                if not isinstance(value, Mapping):
                    raise ValueError("edit operations must be objects.")
                segment_id = str(value.get("segment_id", ""))
                if segment_id not in by_segment:
                    raise ValueError("Unknown audio segment in edit operation.")
                operation = EditOperation.from_mapping(value)
                if operation.id in seen_ids:
                    raise ValueError(f"duplicate operation id: {operation.id!r}")
                seen_ids.add(operation.id)
                by_segment[segment_id].append(operation)
            if not seen_ids and not enhance:
                raise ValueError("edit generation requires at least one operation.")

            compiled_parts = [
                compile_edit(
                    segment.original_transcript,
                    by_segment[segment.id],
                )
                for segment in source.segments
            ]
            source_text = self._join_transcripts(
                [segment.original_transcript for segment in source.segments]
            )
            target_text = self._join_transcripts(
                [compiled.target_text for compiled in compiled_parts]
            )
            separators = [
                self._separator(left.original_transcript, right.original_transcript)
                for left, right in zip(
                    source.segments, source.segments[1:], strict=False
                )
            ]
            instruction = concat_edit_instruction_values(
                [compiled.instruction for compiled in compiled_parts],
                separators=separators,
            )
            if enhance:
                instruction = f"<enhance>{instruction}</enhance>"
            return source_text, CompiledEdit(
                target_text=target_text,
                instruction=instruction,
            )

    @staticmethod
    def _decode_segment(segment: AudioSegment) -> tuple[np.ndarray, AudioSegment]:
        try:
            waveform, sample_rate = sf.read(
                segment.audio_path, dtype="float32", always_2d=True
            )
        except Exception as exc:
            raise ValueError("Audio could not be decoded.") from exc
        if waveform.size == 0 or sample_rate <= 0:
            raise ValueError("Audio must not be empty.")
        mono = waveform.mean(axis=1, dtype=np.float32)
        if not np.isfinite(mono).all():
            raise ValueError("Audio contains non-finite samples.")
        if int(sample_rate) != COMPOSITE_SAMPLE_RATE:
            divisor = np.gcd(int(sample_rate), COMPOSITE_SAMPLE_RATE)
            mono = resample_poly(
                mono,
                COMPOSITE_SAMPLE_RATE // divisor,
                int(sample_rate) // divisor,
            ).astype(np.float32, copy=False)
        enriched = replace(
            segment,
            duration_seconds=float(mono.size) / COMPOSITE_SAMPLE_RATE,
        )
        return np.ascontiguousarray(mono), enriched

    def _compose(
        self,
        session_id: str,
        segments: tuple[AudioSegment, ...],
        text: str,
        *,
        noise_overlay: NoiseOverlay | None = None,
        preset_name: str | None = None,
    ) -> AudioReference:
        if not 1 <= len(segments) <= MAX_SOURCE_SEGMENTS:
            raise ValueError("Source Audio supports 1 to 3 segments.")
        decoded = [self._decode_segment(segment) for segment in segments]
        total = sum(item[0].size for item in decoded) / COMPOSITE_SAMPLE_RATE
        if total > MAX_SOURCE_SECONDS + 1e-6:
            raise ValueError("Source Audio must not exceed 30 seconds.")
        enriched = tuple(item[1] for item in decoded)
        if len(enriched) == 1:
            segment = enriched[0]
            reference = AudioReference(
                segment.audio_path,
                text,
                revision_id=segment.revision_id,
                segments=enriched,
                preset_name=preset_name,
            )
        else:
            directory = (
                self.service.config.output_dir / self._key(session_id) / "composites"
            )
            directory.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha1(
                "\0".join(
                    f"{item.id}:{Path(item.audio_path).stat().st_mtime_ns}"
                    for item in enriched
                ).encode(),
                usedforsecurity=False,
            ).hexdigest()[:20]
            output = directory / f"{digest}.wav"
            if not output.is_file():
                temporary = directory / f".{uuid.uuid4().hex}.wav"
                sf.write(
                    temporary,
                    np.concatenate([item[0] for item in decoded]),
                    COMPOSITE_SAMPLE_RATE,
                )
                temporary.replace(output)
            reference = AudioReference(
                str(output),
                text,
                segments=enriched,
                preset_name=preset_name,
            )
        if noise_overlay is None:
            return reference
        item = NoiseUpload(
            id=noise_overlay.item_id,
            name=noise_overlay.item_id,
            audio_path=noise_overlay.noise_path,
            duration_seconds=self._duration(noise_overlay.noise_path),
            history_consent=noise_overlay.history_consent,
        )
        return self._mix_noise(
            session_id,
            reference,
            item,
            kind=noise_overlay.kind,
            snr_db=noise_overlay.snr_db,
            crop_start=noise_overlay.crop_start,
        )

    @staticmethod
    def _check_version(reference: AudioReference, expected: str) -> None:
        if not expected or expected != reference.composition_version:
            raise ValueError("Source Audio changed; refresh and try again.")

    def _key(self, session_id: str) -> str:
        return _safe_session_id(session_id)

    def lock(self, session_id: str) -> threading.RLock:
        key = self._key(session_id)
        with self._guard:
            return self._locks.setdefault(key, threading.RLock())

    def get(self, session_id: str) -> SessionState:
        key = self._key(session_id)
        with self._guard:
            if key not in self._states:
                prompt = (
                    self._reference(
                        self._default_prompt_preset.audio_path,
                        self._default_prompt_preset.prompt_text,
                    )
                    if self._default_prompt_preset is not None
                    else None
                )
                edit_source = (
                    self._build_edit_source_preset(
                        session_id,
                        self._default_edit_source_preset.name,
                    )
                    if self._default_edit_source_preset is not None
                    else None
                )
                self._states[key] = SessionState(
                    prompt=prompt,
                    edit_source=edit_source,
                    selected_model_id=self.service.default_model_id,
                )
            return self._states[key]

    def set(self, session_id: str, state: SessionState) -> SessionState:
        key = self._key(session_id)
        with self._guard:
            self._states[key] = state
            return state

    def select_model(self, session_id: str, model_id: str) -> SessionState:
        self.service.get_model_spec(model_id)
        with self.lock(session_id):
            return self.set(
                session_id,
                replace(self.get(session_id), selected_model_id=model_id),
            )

    @staticmethod
    def _upload_suffix(filename: str | None) -> str:
        suffix = Path(filename or "audio.wav").suffix.lower()
        return (
            suffix
            if suffix in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}
            else ".wav"
        )

    @staticmethod
    def _store_upload(upload: UploadFile, output: Path) -> None:
        total = 0
        try:
            with output.open("wb") as file_obj:
                while chunk := upload.file.read(1_024 * 1_024):
                    total += len(chunk)
                    if total > MAX_UPLOAD_BYTES:
                        raise ValueError("Audio file must not exceed 8192 KiB.")
                    file_obj.write(chunk)
        except Exception:
            output.unlink(missing_ok=True)
            raise

    @classmethod
    def _validate_uploaded_duration(cls, path: Path) -> float:
        duration = cls._duration(str(path))
        if duration > MAX_AUDIO_SECONDS + 1e-6:
            raise ValueError("Audio must not exceed 30 seconds.")
        return duration

    def create_library_upload(
        self,
        session_id: str,
        upload: UploadFile,
        transcript: str,
        multiple_speakers: bool,
        history_consent: bool,
    ) -> UploadedAudio:
        text = transcript.strip()
        if not text:
            raise ValueError("Transcript must not be empty.")
        validate_literal_text(text, label="Transcript")
        upload_id = uuid.uuid4().hex
        directory = self.service.config.output_dir / self._key(session_id) / "uploads"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{upload_id}{self._upload_suffix(upload.filename)}"
        self._store_upload(upload, output)
        try:
            duration = self._validate_uploaded_duration(output)
            item = UploadedAudio(
                id=upload_id,
                name=Path(upload.filename or "Uploaded audio").name,
                audio_path=str(output),
                transcript=text,
                duration_seconds=duration,
                multiple_speakers=multiple_speakers,
                allow_xvector=not multiple_speakers,
                history_consent=history_consent,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            with self.lock(session_id):
                state = self.get(session_id).add_upload(item)
                self.set(session_id, state)
            return item
        except Exception:
            output.unlink(missing_ok=True)
            raise

    def _library_reference(
        self, session_id: str, kind: str, item_id: str
    ) -> AudioReference:
        state = self.get(session_id)
        if kind == "presets":
            try:
                preset = self._presets[item_id]
            except KeyError as exc:
                raise ValueError("Unknown audio library item.") from exc
            return self._reference(preset.audio_path, preset.prompt_text)
        if kind == "uploads":
            self._require_uploads_enabled()
            return state.find_upload(item_id).as_reference()
        if kind == "revisions":
            revision = state.find_revision(item_id)
            return revision.as_reference()
        raise ValueError("Unknown audio library kind.")

    def list_audio_library(
        self,
        session_id: str,
        kind: str,
        *,
        page: int,
        page_size: int = 10,
    ) -> dict[str, Any]:
        if page < 1:
            raise ValueError("Page must be at least 1.")
        if page_size != 10:
            raise ValueError("Audio library pages contain 10 items.")
        state = self.get(session_id)
        if kind == "presets":
            records = [
                {
                    "id": preset.name,
                    "kind": kind,
                    "title": preset.name,
                    "transcript": preset.prompt_text,
                    "duration_seconds": self._duration(preset.audio_path),
                    "multiple_speakers": False,
                    "created_at": None,
                }
                for preset in self.service.prompt_presets
            ]
        elif kind == "uploads":
            self._require_uploads_enabled()
            records = [
                {
                    "id": item.id,
                    "kind": kind,
                    "title": item.name,
                    "transcript": item.transcript,
                    "duration_seconds": item.duration_seconds,
                    "multiple_speakers": item.multiple_speakers,
                    "created_at": item.created_at,
                }
                for item in reversed(state.uploads)
            ]
        elif kind == "revisions":
            records = [
                {
                    "id": item.id,
                    "kind": kind,
                    "title": "Edit result"
                    if item.metadata.get("kind") == "edit"
                    else "TTS result",
                    "transcript": item.text,
                    "duration_seconds": float(
                        item.metadata.get("duration_seconds", 0.0)
                    ),
                    "multiple_speakers": False,
                    "created_at": item.created_at,
                }
                for item in reversed(state.revisions)
            ]
        else:
            raise ValueError("Unknown audio library kind.")
        total = len(records)
        start = (page - 1) * page_size
        items = records[start : start + page_size]
        for item in items:
            item["audio_url"] = (
                f"/api/session/{self._key(session_id)}/audio-library/{kind}/"
                f"{quote(str(item['id']), safe='')}/audio"
            )
        return {
            "items": items,
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    def apply_audio_library_item(
        self,
        session_id: str,
        *,
        kind: str,
        item_id: str,
        destination: str,
        action: str,
        expected_version: str = "",
        index: int | None = None,
        replace_segment_id: str | None = None,
    ) -> SessionState:
        if destination not in DESTINATIONS:
            raise ValueError("Unknown audio destination.")
        with self.lock(session_id):
            state = self.get(session_id)
            incoming = self._library_reference(session_id, kind, item_id)
            if action == "bind":
                if (
                    destination == "edit_source"
                    and incoming.duration_seconds > MAX_SOURCE_SECONDS
                ):
                    raise ValueError("Source Audio must not exceed 30 seconds.")
                return self.set(session_id, state.with_reference(destination, incoming))
            if destination != "edit_source":
                raise ValueError("Only Edit Source supports segment changes.")
            current = state.edit_source
            if current is None:
                raise ValueError("Choose Source Audio first.")
            self._check_version(current, expected_version)
            if action == "insert":
                if len(current.segments) >= MAX_SOURCE_SEGMENTS:
                    raise ValueError("Source Audio supports at most 3 segments.")
                if index is None or not 0 <= index <= len(current.segments):
                    raise ValueError("Invalid insertion position.")
                segment = replace(incoming.segments[0], id=uuid.uuid4().hex)
                return self._insert_segment_value(
                    session_id, state, current, segment, index
                )
            if action == "replace":
                if not replace_segment_id:
                    raise ValueError("Replacement segment is required.")
                segments = list(current.segments)
                try:
                    segment_index = next(
                        position
                        for position, segment in enumerate(segments)
                        if segment.id == replace_segment_id
                    )
                except StopIteration as exc:
                    raise ValueError("Unknown audio segment.") from exc
                segments[segment_index] = replace(
                    incoming.segments[0], id=uuid.uuid4().hex
                )
                text = self._join_transcripts(
                    [segment.original_transcript for segment in segments]
                )
                reference = self._compose(
                    session_id,
                    tuple(segments),
                    text,
                    noise_overlay=current.noise_overlay,
                )
                return self.set(
                    session_id, state.with_reference("edit_source", reference)
                )
            raise ValueError("Unknown audio library action.")

    def resolve_audio_library_path(
        self, session_id: str, kind: str, item_id: str
    ) -> str:
        return self._library_reference(session_id, kind, item_id).audio_path

    def _noise_items(self, session_id: str, kind: str) -> list[NoiseUpload]:
        if kind == "presets":
            return list(self._noise_presets.values())
        if kind == "uploads":
            self._require_uploads_enabled()
            return list(reversed(self.get(session_id).noise_uploads))
        raise ValueError("unknown noise library kind.")

    def noise_library_page(
        self,
        session_id: str,
        kind: str,
        *,
        page: int = 1,
        page_size: int = 10,
        selected_id: str | None = None,
    ) -> dict[str, Any]:
        if page < 1 or page_size != 10:
            raise ValueError("Noise Library uses pages of 10 items.")
        items = self._noise_items(session_id, kind)
        total = len(items)
        total_pages = max(1, (total + page_size - 1) // page_size)
        page = min(page, total_pages)
        visible = items[(page - 1) * page_size : page * page_size]
        return {
            "items": [
                {
                    "id": item.id,
                    "kind": kind,
                    "title": item.name,
                    "audio_url": (
                        f"/api/session/{self._key(session_id)}/noise-library/"
                        f"{kind}/{quote(item.id, safe='')}/audio"
                    ),
                    "duration_seconds": item.duration_seconds,
                    "created_at": item.created_at or None,
                }
                for item in visible
            ],
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": total_pages,
            "selected_id": selected_id,
        }

    def _noise_item(
        self,
        session_id: str,
        kind: str,
        item_id: str,
    ) -> NoiseUpload:
        if kind == "presets":
            try:
                return self._noise_presets[item_id]
            except KeyError as exc:
                raise ValueError("Unknown noise preset.") from exc
        if kind == "uploads":
            self._require_uploads_enabled()
            try:
                return self.get(session_id).find_noise_upload(item_id)
            except ValueError as exc:
                raise ValueError("Unknown noise upload.") from exc
        raise ValueError("unknown noise library kind.")

    def create_noise_upload(
        self,
        session_id: str,
        upload: UploadFile,
        history_consent: bool,
    ) -> dict[str, Any]:
        suffix = Path(upload.filename or "noise.wav").suffix.lower()
        if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}:
            suffix = ".wav"
        directory = (
            self.service.config.output_dir / self._key(session_id) / "noise-uploads"
        )
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{uuid.uuid4().hex}{suffix}"
        self._store_upload(upload, output)
        try:
            duration = self._validate_uploaded_duration(output)
            item = NoiseUpload(
                id=uuid.uuid4().hex,
                name=Path(upload.filename or "Uploaded noise").stem,
                audio_path=str(output),
                duration_seconds=duration,
                history_consent=history_consent,
            )
            with self.lock(session_id):
                self.set(session_id, self.get(session_id).add_noise_upload(item))
            return self.noise_library_page(
                session_id,
                "uploads",
                selected_id=item.id,
            )
        except Exception:
            output.unlink(missing_ok=True)
            raise

    @staticmethod
    def _read_mono_48khz(path: str) -> torch.Tensor:
        try:
            waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise ValueError("Audio could not be decoded.") from exc
        if waveform.size == 0 or int(sample_rate) <= 0:
            raise ValueError("Audio must not be empty.")
        mono = torch.from_numpy(waveform.mean(axis=1, dtype=np.float32))
        if not bool(torch.isfinite(mono).all()):
            raise ValueError("Audio contains non-finite samples.")
        if int(sample_rate) != COMPOSITE_SAMPLE_RATE:
            mono = high_quality_resample(
                mono.unsqueeze(0),
                orig_sr=int(sample_rate),
                target_sr=COMPOSITE_SAMPLE_RATE,
            ).squeeze(0)
        return mono.contiguous()

    @staticmethod
    def _noise_crop_fraction(
        reference: AudioReference,
        item: NoiseUpload,
        length: int,
    ) -> float:
        digest = hashlib.sha1(
            (
                f"{reference.clean_audio_path}\0{reference.text}\0{item.id}\0{length}"
            ).encode(),
            usedforsecurity=False,
        ).digest()
        return int.from_bytes(digest[:8], "big") / float((1 << 64) - 1)

    def _mix_noise(
        self,
        session_id: str,
        reference: AudioReference,
        item: NoiseUpload,
        *,
        kind: str,
        snr_db: int,
        crop_start: int | None = None,
    ) -> AudioReference:
        clean_path = reference.clean_audio_path or reference.audio_path
        clean = self._read_mono_48khz(clean_path)
        noise = self._read_mono_48khz(item.audio_path)
        max_start = max(0, int(noise.numel()) - int(clean.numel()))
        crop_fraction = (
            self._noise_crop_fraction(reference, item, int(clean.numel()))
            if crop_start is None
            else min(max(0, int(crop_start)), max_start) / float(max_start + 1)
        )
        fitted, details = fit_noise_segment(
            noise,
            length=int(clean.numel()),
            sample_rate=COMPOSITE_SAMPLE_RATE,
            crop_fraction=crop_fraction,
            fade_ms=10.0,
            random_crop_after_tiling=True,
        )
        mixed, _ = mix_region_at_snr(clean, fitted, snr_db=snr_db)
        mixed, _ = limit_peak(
            mixed,
            effective_length=int(mixed.numel()),
            peak_limit=0.99,
        )
        directory = (
            self.service.config.output_dir / self._key(session_id) / "noise-mixes"
        )
        directory.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha1(
            (
                f"{clean_path}\0{Path(clean_path).stat().st_mtime_ns}\0"
                f"{item.id}\0{snr_db}\0{details['crop_start']}"
            ).encode(),
            usedforsecurity=False,
        ).hexdigest()[:20]
        output = directory / f"{digest}.wav"
        if not output.is_file():
            temporary = directory / f".{uuid.uuid4().hex}.wav"
            sf.write(temporary, mixed.cpu().numpy(), COMPOSITE_SAMPLE_RATE)
            temporary.replace(output)
        return AudioReference(
            audio_path=str(output),
            clean_audio_path=clean_path,
            noise_overlay=NoiseOverlay(
                item_id=item.id,
                kind=kind,
                noise_path=item.audio_path,
                snr_db=int(snr_db),
                crop_start=int(details["crop_start"]),
                tiled=bool(details["tiled"]),
                history_consent=item.history_consent,
            ),
            text=reference.text,
            revision_id=reference.revision_id,
            segments=reference.segments,
            preset_name=reference.preset_name,
        )

    def set_source_noise(
        self,
        session_id: str,
        *,
        kind: str,
        item_id: str,
        snr_db: int,
        expected_version: str,
        crop_start: int | None = None,
    ) -> SessionState:
        if not isinstance(snr_db, int) or not 0 <= snr_db <= 20:
            raise ValueError("Noise SNR must be an integer between 0 and 20 dB.")
        with self.lock(session_id):
            state = self.get(session_id)
            source = state.edit_source
            if source is None:
                raise ValueError("Choose Source Audio first.")
            self._check_version(source, expected_version)
            item = self._noise_item(session_id, kind, item_id)
            reference = self._mix_noise(
                session_id,
                source,
                item,
                kind=kind,
                snr_db=snr_db,
                crop_start=crop_start,
            )
            return self.set(
                session_id,
                state.with_reference("edit_source", reference),
            )

    def clear_source_noise(
        self,
        session_id: str,
        *,
        expected_version: str,
    ) -> SessionState:
        with self.lock(session_id):
            state = self.get(session_id)
            source = state.edit_source
            if source is None:
                raise ValueError("Choose Source Audio first.")
            self._check_version(source, expected_version)
            reference = AudioReference(
                audio_path=source.clean_audio_path or source.audio_path,
                clean_audio_path=source.clean_audio_path or source.audio_path,
                text=source.text,
                revision_id=source.revision_id,
                segments=source.segments,
                preset_name=source.preset_name,
            )
            return self.set(
                session_id,
                state.with_reference("edit_source", reference),
            )

    def bind_upload(
        self,
        session_id: str,
        destination: str,
        upload: UploadFile,
        transcript: str,
        multiple_speakers: bool,
        history_consent: bool,
    ) -> SessionState:
        if destination not in DESTINATIONS:
            raise ValueError(f"unknown destination: {destination}")
        suffix = Path(upload.filename or "audio.wav").suffix.lower()
        if suffix not in {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus"}:
            suffix = ".wav"
        directory = self.service.config.output_dir / self._key(session_id) / "uploads"
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / f"{uuid.uuid4().hex}{suffix}"
        self._store_upload(upload, output)
        try:
            self._validate_uploaded_duration(output)
            reference = self._reference(
                str(output),
                transcript,
                multiple_speakers=multiple_speakers,
                history_consent=history_consent,
            )
            if (
                destination == "edit_source"
                and reference.duration_seconds > MAX_SOURCE_SECONDS
            ):
                raise ValueError("Source Audio must not exceed 30 seconds.")
            with self.lock(session_id):
                return self.set(
                    session_id,
                    self.get(session_id).with_reference(destination, reference),
                )
        except Exception:
            output.unlink(missing_ok=True)
            raise

    def use_preset(
        self, session_id: str, preset_name: str, destination: str = "prompt"
    ) -> SessionState:
        if destination not in DESTINATIONS:
            raise ValueError(f"unknown destination: {destination}")
        if destination == "edit_source" and preset_name in EDIT_SOURCE_PRESET_BY_NAME:
            reference = self._build_edit_source_preset(session_id, preset_name)
        else:
            try:
                preset = self._presets[preset_name]
            except KeyError as exc:
                raise ValueError(f"unknown voice preset: {preset_name}") from exc
            reference = self._reference(preset.audio_path, preset.prompt_text)
        if (
            destination == "edit_source"
            and reference.duration_seconds > MAX_SOURCE_SECONDS
        ):
            raise ValueError("Source Audio must not exceed 30 seconds.")
        with self.lock(session_id):
            return self.set(
                session_id, self.get(session_id).with_reference(destination, reference)
            )

    def _build_edit_source_preset(
        self,
        session_id: str,
        preset_name: str,
    ) -> AudioReference:
        try:
            spec = EDIT_SOURCE_PRESET_BY_NAME[preset_name]
        except KeyError as exc:
            raise ValueError(f"unknown edit source preset: {preset_name}") from exc
        segments: list[AudioSegment] = []
        for index, source_name in enumerate(spec.segment_presets):
            try:
                source_preset = self._presets[source_name]
            except KeyError as exc:
                raise ValueError(
                    f"edit source preset {preset_name!r} requires {source_name!r}"
                ) from exc
            segment = self._reference(
                source_preset.audio_path,
                source_preset.prompt_text,
                segment_id=f"preset-{preset_name}-{index + 1}",
            ).segments[0]
            segments.append(segment)
        text = self._join_transcripts(
            [segment.original_transcript for segment in segments]
        )
        reference = self._compose(
            session_id,
            tuple(segments),
            text,
            preset_name=preset_name,
        )
        if spec.noise is None:
            return reference
        try:
            noise = self._noise_presets[spec.noise.item_id]
        except KeyError as exc:
            raise ValueError(
                f"edit source preset {preset_name!r} requires a missing noise preset"
            ) from exc
        return self._mix_noise(
            session_id,
            reference,
            noise,
            kind="presets",
            snr_db=spec.noise.snr_db,
            crop_start=spec.noise.crop_start,
        )

    def route(self, session_id: str, source: str, destination: str) -> SessionState:
        with self.lock(session_id):
            state = self.get(session_id)
            reference = state.resolve_reference(source)
            if reference.duration_seconds <= 0:
                reference = self._reference(
                    reference.audio_path,
                    reference.text,
                    revision_id=reference.revision_id,
                    allow_xvector=reference.use_xvector,
                )
            if (
                destination == "edit_source"
                and reference.duration_seconds > MAX_SOURCE_SECONDS
            ):
                raise ValueError("Source Audio must not exceed 30 seconds.")
            if destination == "prompt" and len(reference.segments) > 1:
                reference = self._reference(
                    reference.audio_path,
                    reference.text,
                    allow_xvector=False,
                )
            return self.set(session_id, state.with_reference(destination, reference))

    def insert_segment(
        self,
        session_id: str,
        source: str,
        index: int,
        expected_version: str,
    ) -> SessionState:
        with self.lock(session_id):
            state = self.get(session_id)
            current = state.edit_source
            if current is None:
                raise ValueError("Choose Source Audio first.")
            self._check_version(current, expected_version)
            if len(current.segments) >= MAX_SOURCE_SEGMENTS:
                raise ValueError("Source Audio supports at most 3 segments.")
            if not 0 <= index <= len(current.segments):
                raise ValueError("Invalid insertion position.")
            incoming = state.resolve_reference(source)
            if source == "edit_source" or len(incoming.segments) != 1:
                raise ValueError("Insert one audio segment at a time.")
            segment = replace(incoming.segments[0], id=uuid.uuid4().hex)
            return self._insert_segment_value(
                session_id, state, current, segment, index
            )

    def _insert_segment_value(
        self,
        session_id: str,
        state: SessionState,
        current: AudioReference,
        segment: AudioSegment,
        index: int,
    ) -> SessionState:
        segments = current.segments[:index] + (segment,) + current.segments[index:]
        text = self._join_transcripts([item.original_transcript for item in segments])
        reference = self._compose(
            session_id,
            segments,
            text,
            noise_overlay=current.noise_overlay,
        )
        return self.set(session_id, state.with_reference("edit_source", reference))

    def insert_uploaded_segment(
        self,
        session_id: str,
        upload: UploadFile,
        transcript: str,
        multiple_speakers: bool,
        history_consent: bool,
        index: int,
        expected_version: str,
    ) -> SessionState:
        directory = self.service.config.output_dir / self._key(session_id) / "uploads"
        directory.mkdir(parents=True, exist_ok=True)
        suffix = Path(upload.filename or "audio.wav").suffix.lower()
        output = directory / f"{uuid.uuid4().hex}{suffix if suffix else '.wav'}"
        self._store_upload(upload, output)
        try:
            self._validate_uploaded_duration(output)
            temp_reference = self._reference(
                str(output),
                transcript,
                multiple_speakers=multiple_speakers,
                history_consent=history_consent,
            )
            with self.lock(session_id):
                state = self.get(session_id)
                current = state.edit_source
                if current is None:
                    raise ValueError("Choose Source Audio first.")
                self._check_version(current, expected_version)
                if len(current.segments) >= MAX_SOURCE_SEGMENTS:
                    raise ValueError("Source Audio supports at most 3 segments.")
                if not 0 <= index <= len(current.segments):
                    raise ValueError("Invalid insertion position.")
                segment = replace(temp_reference.segments[0], id=uuid.uuid4().hex)
                return self._insert_segment_value(
                    session_id, state, current, segment, index
                )
        except Exception:
            output.unlink(missing_ok=True)
            raise

    def replace_uploaded_segment(
        self,
        session_id: str,
        segment_id: str,
        upload: UploadFile,
        transcript: str,
        multiple_speakers: bool,
        history_consent: bool,
        expected_version: str,
    ) -> SessionState:
        directory = self.service.config.output_dir / self._key(session_id) / "uploads"
        directory.mkdir(parents=True, exist_ok=True)
        suffix = Path(upload.filename or "audio.wav").suffix.lower()
        output = directory / f"{uuid.uuid4().hex}{suffix if suffix else '.wav'}"
        self._store_upload(upload, output)
        try:
            self._validate_uploaded_duration(output)
            incoming = self._reference(
                str(output),
                transcript,
                multiple_speakers=multiple_speakers,
                history_consent=history_consent,
            ).segments[0]
            with self.lock(session_id):
                state = self.get(session_id)
                current = state.edit_source
                if current is None:
                    raise ValueError("Choose Source Audio first.")
                self._check_version(current, expected_version)
                segments = list(current.segments)
                try:
                    index = next(
                        i for i, item in enumerate(segments) if item.id == segment_id
                    )
                except StopIteration as exc:
                    raise ValueError("Unknown audio segment.") from exc
                segments[index] = replace(incoming, id=uuid.uuid4().hex)
                text = self._join_transcripts(
                    [item.original_transcript for item in segments]
                )
                reference = self._compose(
                    session_id,
                    tuple(segments),
                    text,
                    noise_overlay=current.noise_overlay,
                )
                updated = self.set(
                    session_id, state.with_reference("edit_source", reference)
                )
                return updated
        except Exception:
            output.unlink(missing_ok=True)
            raise

    def mutate_segment(
        self,
        session_id: str,
        segment_id: str,
        *,
        action: str,
        expected_version: str,
        to_index: int | None = None,
        source: str | None = None,
    ) -> SessionState:
        with self.lock(session_id):
            state = self.get(session_id)
            current = state.edit_source
            if current is None:
                raise ValueError("Choose Source Audio first.")
            self._check_version(current, expected_version)
            segments = list(current.segments)
            try:
                old_index = next(
                    i for i, item in enumerate(segments) if item.id == segment_id
                )
            except StopIteration as exc:
                raise ValueError("Unknown audio segment.") from exc
            if action == "delete":
                if len(segments) == 1:
                    raise ValueError("Source Audio must keep at least one segment.")
                segments.pop(old_index)
            elif action == "move":
                if to_index is None or not 0 <= to_index <= len(segments):
                    raise ValueError("Invalid move position.")
                segment = segments.pop(old_index)
                adjusted = to_index - 1 if to_index > old_index else to_index
                if adjusted == old_index:
                    return state
                segments.insert(adjusted, segment)
            elif action == "replace":
                if not source:
                    raise ValueError("Replacement source is required.")
                incoming = state.resolve_reference(source)
                if len(incoming.segments) != 1:
                    raise ValueError("Replace with one audio segment.")
                segments[old_index] = replace(incoming.segments[0], id=uuid.uuid4().hex)
            else:
                raise ValueError("Unknown segment action.")
            text = self._join_transcripts(
                [item.original_transcript for item in segments]
            )
            reference = self._compose(
                session_id,
                tuple(segments),
                text,
                noise_overlay=current.noise_overlay,
            )
            return self.set(session_id, state.with_reference("edit_source", reference))

    def update_segment_transcript(
        self,
        session_id: str,
        segment_id: str,
        transcript: str,
        expected_version: str,
    ) -> SessionState:
        text = transcript.strip()
        if not text:
            raise ValueError("Source Transcript must not be empty.")
        validate_literal_text(text, label="Source Transcript")
        with self.lock(session_id):
            state = self.get(session_id)
            current = state.edit_source
            if current is None:
                raise ValueError("Choose Source Audio first.")
            self._check_version(current, expected_version)
            segments = list(current.segments)
            try:
                index = next(
                    i for i, segment in enumerate(segments) if segment.id == segment_id
                )
            except StopIteration as exc:
                raise ValueError("Unknown audio segment.") from exc
            segments[index] = replace(segments[index], original_transcript=text)
            aggregate = self._join_transcripts(
                [segment.original_transcript for segment in segments]
            )
            reference = AudioReference(
                audio_path=current.audio_path,
                text=aggregate,
                revision_id=current.revision_id if len(segments) == 1 else None,
                segments=tuple(segments),
                clean_audio_path=current.clean_audio_path,
                noise_overlay=current.noise_overlay,
            )
            return self.set(
                session_id,
                state.with_reference("edit_source", reference),
            )

    def clear_reference(self, session_id: str, destination: str) -> SessionState:
        if destination not in DESTINATIONS:
            raise ValueError(f"unknown destination: {destination}")
        with self.lock(session_id):
            return self.set(
                session_id, self.get(session_id).with_reference(destination, None)
            )

    def update_text(self, session_id: str, destination: str, text: str) -> SessionState:
        if destination == "edit_source":
            raise ValueError("Edit Source transcripts must be updated per segment.")
        with self.lock(session_id):
            return self.set(
                session_id,
                self.get(session_id).set_reference_text(destination, text),
            )

    def add_result(self, session_id: str, result: GenerationResult) -> RevisionNode:
        state = self.get(session_id)
        kind = str(result.metadata.get("kind", ""))
        prompt = state.prompt if kind == "tts" else None
        source = state.edit_source if kind == "edit" else None
        consent = history_allowed(prompt) and history_allowed(source)
        revision_id = uuid.uuid4().hex
        metadata = dict(result.metadata)
        metadata["history_consent"] = consent
        if self.history_store is not None and consent:
            try:
                self.history_store.record_success(
                    event_id=revision_id,
                    session_id=self._key(session_id),
                    result=result,
                    prompt=prompt,
                    source=source,
                )
            except Exception as exc:
                print(f"History record failed for {revision_id}: {exc}", flush=True)
            else:
                metadata["history_event_id"] = revision_id
        revision = RevisionNode.create(
            revision_id=revision_id,
            audio_path=result.audio_path,
            text=result.text,
            parent_id=result.parent_id,
            metadata=metadata,
        )
        with self.lock(session_id):
            previous = self.get(session_id)
            retention = self.service.config.output_retention_count
            current = previous.add_revision(
                revision,
                max_revisions=(
                    retention if retention > 0 else len(previous.revisions) + 1
                ),
            )
            self.set(session_id, current)
            remove_pruned_audio(
                previous,
                current,
                managed_root=self.service.config.output_dir,
            )
        return revision

    def resolve_audio(self, session_id: str, source: str) -> AudioReference:
        return self.get(session_id).resolve_reference(source)

    def _reference_json(
        self, session_id: str, source: str, reference: AudioReference | None
    ) -> dict[str, Any] | None:
        if reference is None:
            return None
        version = reference.composition_version
        preset_name = reference.preset_name or next(
            (
                preset.name
                for preset in self.service.prompt_presets
                if preset.audio_path == reference.audio_path
                and preset.prompt_text == reference.text
            ),
            None,
        )
        return {
            "source_id": source,
            "audio_url": f"/api/audio/{self._key(session_id)}/{source}?v={version}",
            "text": reference.text,
            "revision_id": reference.revision_id,
            "duration_seconds": reference.duration_seconds,
            "composition_version": reference.composition_version,
            "use_xvector": reference.use_xvector,
            "noise": (
                {
                    "item_id": reference.noise_overlay.item_id,
                    "kind": reference.noise_overlay.kind,
                    "snr_db": reference.noise_overlay.snr_db,
                }
                if reference.noise_overlay is not None
                else None
            ),
            "segments": [
                {
                    "id": segment.id,
                    "source_id": segment.id,
                    "audio_url": (
                        f"/api/audio/{self._key(session_id)}/{segment.id}"
                        f"?v={hashlib.sha1(segment.audio_path.encode(), usedforsecurity=False).hexdigest()[:10]}"
                    ),
                    "transcript": segment.original_transcript,
                    "duration_seconds": segment.duration_seconds,
                    "revision_id": segment.revision_id,
                    "multiple_speakers": segment.multiple_speakers,
                    "allow_xvector": segment.allow_xvector,
                }
                for segment in reference.segments
            ],
            "origin": (
                {"kind": "revision", "name": None}
                if reference.revision_id
                else {"kind": "preset", "name": preset_name}
                if preset_name
                else {"kind": "custom", "name": None}
            ),
        }

    def snapshot(self, session_id: str) -> dict[str, Any]:
        state = self.get(session_id)
        revisions = [
            {
                "id": node.id,
                "audio_url": f"/api/audio/{self._key(session_id)}/{node.id}",
                "text": node.text,
                "parent_id": node.parent_id,
                "created_at": node.created_at,
                "metadata": _plain(node.metadata),
            }
            for node in reversed(state.revisions)
        ]
        return {
            "session_id": self._key(session_id),
            "selected_model_id": state.selected_model_id
            or self.service.default_model_id,
            "models": self.service.model_catalog(),
            "prompt": self._reference_json(session_id, "prompt", state.prompt),
            "edit_source": self._reference_json(
                session_id, "edit_source", state.edit_source
            ),
            "revisions": revisions,
            "latest": revisions[0] if revisions else None,
            "presets": [
                {"name": item.name, "text": item.prompt_text}
                for item in self.service.prompt_presets
            ],
            "prompt_presets": [
                {"name": item.name, "text": item.prompt_text}
                for item in self._prompt_presets
            ],
            "edit_source_presets": [
                {
                    "name": item.name,
                    "text": self._join_transcripts(
                        [
                            self._presets[name].prompt_text
                            for name in item.segment_presets
                        ]
                    ),
                }
                for item in self._edit_source_preset_specs
            ],
            "defaults": {
                "target_text": DEFAULT_TARGET_TEXT,
                "edit_operations": self._starter_operations(state.edit_source),
                "enhance": bool(
                    state.edit_source
                    and state.edit_source.preset_name
                    and EDIT_SOURCE_PRESET_BY_NAME[
                        state.edit_source.preset_name
                    ].enhance
                ),
            },
            "ode_methods": list(SUPPORTED_ODE_METHODS),
        }

    def _starter_operations(
        self, reference: AudioReference | None
    ) -> list[dict[str, Any]]:
        if reference is None:
            return []
        preset_name = reference.preset_name
        if preset_name not in EDIT_SOURCE_PRESET_BY_NAME:
            return []
        operations = build_segmented_preset_operations(
            preset_name,
            [segment.original_transcript for segment in reference.segments],
        )
        return [
            {
                **{
                    key: value
                    for key, value in operation.items()
                    if key != "segment_index"
                },
                "segment_id": reference.segments[int(operation["segment_index"])].id,
            }
            for operation in operations
        ]


def create_studio_app(
    service: StudioService,
    *,
    frontend_dist: Path,
    startup_state: StartupState | None = None,
    recognition_jobs: RecognitionJobStore | None = None,
    gpu_recognition_handler: Callable[[str], str] | None = None,
    history_store: GenerationHistoryStore | None = None,
    uploads_enabled: bool = True,
) -> tuple[gr.Server, StudioSessionStore]:
    app = gr.Server(title="dots.tts.edit", docs_url=None, redoc_url=None)
    store = StudioSessionStore(
        service,
        history_store=history_store,
        uploads_enabled=uploads_enabled,
    )

    if not uploads_enabled:

        @app.middleware("http")
        async def block_gradio_uploads(
            request: Request, call_next: Callable
        ) -> Response:
            if request.url.path.startswith("/gradio_api/upload"):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Uploads are disabled for this deployment."},
                )
            return await call_next(request)

    def fail(exc: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(exc))

    readiness = startup_state or StartupState(frontend="ready", model="ready")

    @app.get("/api/healthz")
    def healthz() -> dict[str, Any]:
        status = readiness.snapshot()
        status.update(
            {
                "optimize": service.config.optimize,
                "precision": service.config.precision,
                "runtime_loaded": service.runtime_loaded,
                "runtime_loaded_count": sum(
                    bool(item["runtime_loaded"]) for item in service.model_catalog()
                ),
                "default_model_id": service.default_model_id,
                "models": service.model_catalog(),
                "max_generate_length": service.config.max_generate_length,
                "recognition_available": (
                    uploads_enabled and recognition_jobs is not None
                ),
                "uploads_enabled": uploads_enabled,
                "demo_url": os.environ.get(
                    "DOTS_TTS_DEMO_URL",
                    "https://huggingface.co/spaces/dots-studio/dots.tts.edit-demo",
                ),
                "paper_url": os.environ.get(
                    "DOTS_TTS_PAPER_URL",
                    "https://arxiv.org/abs/2608.02673",
                ),
            }
        )
        return status

    @app.put("/api/session/{session_id}/model")
    def select_model(
        session_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        try:
            store.select_model(session_id, str(payload.get("model_id", "")))
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.get("/")
    def index() -> Response:
        status = readiness.snapshot()
        if status["frontend"] == "ready" and (frontend_dist / "index.html").is_file():
            return FileResponse(frontend_dist / "index.html")
        error = status.get("frontend_error")
        message = (
            f"Frontend failed to build: {error}"
            if error
            else "Preparing dots.tts.edit…"
        )
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
            "<title>dots.tts.edit</title><style>body{margin:0;display:grid;place-items:center;"
            "min-height:100vh;background:#f3f3f1;color:#30312f;font:14px -apple-system,BlinkMacSystemFont,sans-serif}"
            ".boot{padding:28px 32px;border:1px solid #deded9;border-radius:14px;background:white}"
            "i{display:inline-block;width:8px;height:8px;margin-right:10px;border-radius:50%;background:#6657c8}</style>"
            f"<div class=boot><i></i>{message}</div>"
            + (
                ""
                if error
                else "<script>setTimeout(()=>location.reload(),1500)</script>"
            ),
            status_code=503 if error else 200,
        )

    @app.get("/assets/{asset_path:path}")
    def asset(asset_path: str) -> FileResponse:
        root = (frontend_dist / "assets").resolve()
        path = (root / asset_path).resolve()
        if root not in path.parents or not path.is_file():
            raise HTTPException(status_code=404)
        return FileResponse(path)

    @app.delete("/api/session/{session_id}/reference/{destination}")
    def clear_reference(session_id: str, destination: str) -> dict[str, Any]:
        try:
            store.clear_reference(session_id, destination)
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.get("/api/session/{session_id}")
    def snapshot(session_id: str) -> dict[str, Any]:
        try:
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.get("/api/session/{session_id}/audio-library/{kind}")
    def audio_library(
        session_id: str,
        kind: str,
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        try:
            return store.list_audio_library(
                session_id, kind, page=page, page_size=page_size
            )
        except ValueError as exc:
            raise fail(exc) from exc

    @app.get("/api/session/{session_id}/audio-library/{kind}/{item_id}/audio")
    def audio_library_file(session_id: str, kind: str, item_id: str) -> FileResponse:
        try:
            return FileResponse(
                store.resolve_audio_library_path(session_id, kind, item_id)
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    if uploads_enabled:

        @app.post("/api/session/{session_id}/audio-library/uploads")
        def create_audio_library_upload(
            session_id: str,
            audio: UploadFile,
            transcript: str = Form(...),
            multiple_speakers: bool = Form(False),
            history_consent: bool = Form(False),
        ) -> dict[str, Any]:
            try:
                item = store.create_library_upload(
                    session_id,
                    audio,
                    transcript,
                    multiple_speakers,
                    history_consent,
                )
                page = store.list_audio_library(
                    session_id, "uploads", page=1, page_size=10
                )
                page["selected_id"] = item.id
                return page
            except ValueError as exc:
                raise fail(exc) from exc

    @app.post("/api/session/{session_id}/audio-library/apply")
    def apply_audio_library(
        session_id: str, payload: dict[str, Any] = Body(...)
    ) -> dict[str, Any]:
        try:
            store.apply_audio_library_item(
                session_id,
                kind=str(payload.get("kind", "")),
                item_id=str(payload.get("item_id", "")),
                destination=str(payload.get("destination", "")),
                action=str(payload.get("action", "")),
                expected_version=str(payload.get("expected_version", "")),
                index=(
                    int(payload["index"]) if payload.get("index") is not None else None
                ),
                replace_segment_id=(
                    str(payload["replace_segment_id"])
                    if payload.get("replace_segment_id") is not None
                    else None
                ),
            )
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.get("/api/session/{session_id}/noise-library/{kind}")
    def noise_library(
        session_id: str,
        kind: str,
        page: int = 1,
        page_size: int = 10,
    ) -> dict[str, Any]:
        try:
            return store.noise_library_page(
                session_id,
                kind,
                page=page,
                page_size=page_size,
            )
        except ValueError as exc:
            raise fail(exc) from exc

    @app.get("/api/session/{session_id}/noise-library/{kind}/{item_id}/audio")
    def noise_library_audio(
        session_id: str,
        kind: str,
        item_id: str,
    ) -> FileResponse:
        try:
            item = store._noise_item(session_id, kind, item_id)  # noqa: SLF001
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(item.audio_path)

    if uploads_enabled:

        @app.post("/api/session/{session_id}/noise-library/uploads")
        def create_noise_upload(
            session_id: str,
            audio: UploadFile,
            history_consent: bool = Form(False),
        ) -> dict[str, Any]:
            try:
                return store.create_noise_upload(session_id, audio, history_consent)
            except ValueError as exc:
                raise fail(exc) from exc

    @app.put("/api/session/{session_id}/edit-source/noise")
    def set_source_noise(
        session_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        try:
            store.set_source_noise(
                session_id,
                kind=str(payload.get("kind", "")),
                item_id=str(payload.get("item_id", "")),
                snr_db=payload.get("snr_db"),
                expected_version=str(payload.get("expected_version", "")),
            )
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.delete("/api/session/{session_id}/edit-source/noise")
    def clear_source_noise(
        session_id: str,
        expected_version: str,
    ) -> dict[str, Any]:
        try:
            store.clear_source_noise(
                session_id,
                expected_version=expected_version,
            )
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    if uploads_enabled:

        @app.post("/api/session/{session_id}/upload/{destination}")
        def upload(
            session_id: str,
            destination: str,
            audio: UploadFile,
            transcript: str = Form(...),
            multiple_speakers: bool = Form(False),
            history_consent: bool = Form(False),
        ) -> dict[str, Any]:
            try:
                store.bind_upload(
                    session_id,
                    destination,
                    audio,
                    transcript,
                    multiple_speakers,
                    history_consent,
                )
                return store.snapshot(session_id)
            except ValueError as exc:
                raise fail(exc) from exc

    @app.post("/api/session/{session_id}/edit-source/segments")
    def insert_segment(
        session_id: str, payload: dict[str, Any] = Body(...)
    ) -> dict[str, Any]:
        try:
            store.insert_segment(
                session_id,
                str(payload.get("source_id", "")),
                int(payload.get("index", -1)),
                str(payload.get("expected_version", "")),
            )
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    if uploads_enabled:

        @app.post("/api/session/{session_id}/edit-source/segments/upload")
        def upload_segment(
            session_id: str,
            audio: UploadFile,
            transcript: str = Form(...),
            multiple_speakers: bool = Form(False),
            history_consent: bool = Form(False),
            expected_version: str = Form(...),
            index: int | None = Form(None),
            replace_segment_id: str | None = Form(None),
        ) -> dict[str, Any]:
            try:
                if replace_segment_id:
                    store.replace_uploaded_segment(
                        session_id,
                        replace_segment_id,
                        audio,
                        transcript,
                        multiple_speakers,
                        history_consent,
                        expected_version,
                    )
                elif index is not None:
                    store.insert_uploaded_segment(
                        session_id,
                        audio,
                        transcript,
                        multiple_speakers,
                        history_consent,
                        index,
                        expected_version,
                    )
                else:
                    raise ValueError("Segment insertion position is required.")
                return store.snapshot(session_id)
            except ValueError as exc:
                raise fail(exc) from exc

    @app.patch("/api/session/{session_id}/edit-source/segments/{segment_id}")
    def mutate_segment(
        session_id: str,
        segment_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        try:
            store.mutate_segment(
                session_id,
                segment_id,
                action=str(payload.get("action", "move")),
                expected_version=str(payload.get("expected_version", "")),
                to_index=(
                    int(payload["to_index"])
                    if payload.get("to_index") is not None
                    else None
                ),
                source=(
                    str(payload["source_id"])
                    if payload.get("source_id") is not None
                    else None
                ),
            )
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.delete("/api/session/{session_id}/edit-source/segments/{segment_id}")
    def delete_segment(
        session_id: str,
        segment_id: str,
        expected_version: str,
    ) -> dict[str, Any]:
        try:
            store.mutate_segment(
                session_id,
                segment_id,
                action="delete",
                expected_version=expected_version,
            )
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.patch("/api/session/{session_id}/edit-source/segments/{segment_id}/transcript")
    def update_segment_transcript(
        session_id: str,
        segment_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        try:
            store.update_segment_transcript(
                session_id,
                segment_id,
                str(payload.get("transcript", "")),
                str(payload.get("expected_version", "")),
            )
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.post("/api/session/{session_id}/preset")
    def preset(session_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            store.use_preset(
                session_id,
                str(payload.get("name", "")),
                str(payload.get("destination", "prompt")),
            )
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.post("/api/session/{session_id}/route")
    def route(session_id: str, payload: dict[str, Any] = Body(...)) -> dict[str, Any]:
        try:
            store.route(
                session_id,
                str(payload.get("source", "")),
                str(payload.get("destination", "")),
            )
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.patch("/api/session/{session_id}/text/{destination}")
    def update_text(
        session_id: str,
        destination: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, Any]:
        try:
            store.update_text(session_id, destination, str(payload.get("text", "")))
            return store.snapshot(session_id)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.post("/api/session/{session_id}/compile-edit")
    def compile_preview(
        session_id: str,
        payload: dict[str, Any] = Body(...),
    ) -> dict[str, str]:
        try:
            _, compiled = store.compile_segment_edit(
                session_id,
                list(payload.get("operations") or []),
                str(payload.get("expected_version", "")),
                enhance=_optional_bool(payload, "enhance"),
            )
            return asdict(compiled)
        except ValueError as exc:
            raise fail(exc) from exc

    @app.get("/api/audio/{session_id}/{source}")
    def audio(session_id: str, source: str) -> FileResponse:
        try:
            reference = store.resolve_audio(session_id, source)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return FileResponse(reference.audio_path)

    def prepare_tts_request(
        session_id: str,
        text: str,
        payload: Mapping[str, Any],
    ) -> TTSRequest:
        settings = _settings(payload)
        state = store.get(session_id)
        prompt = state.prompt
        model_id = str(
            payload.get("model_id")
            or state.selected_model_id
            or service.default_model_id
        )
        service.get_model_spec(model_id)
        use_xvector = _use_xvector(
            payload,
            allowed=prompt.use_xvector if prompt else False,
            default=prompt.use_xvector if prompt else False,
        )
        return TTSRequest(
            text=text,
            prompt_audio_path=prompt.audio_path if prompt else None,
            prompt_text=prompt.text if prompt else None,
            prompt_revision_id=prompt.revision_id if prompt else None,
            model_id=model_id,
            use_xvector=use_xvector,
            settings=settings,
        )

    def prepare_edit_request(
        session_id: str,
        operations: list[Mapping[str, Any]],
        expected_version: str,
        payload: Mapping[str, Any],
    ) -> EditRequest:
        state = store.get(session_id)
        source = state.edit_source
        if source is None:
            raise ValueError("Choose an Edit Source audio first.")
        source_text, compiled = store.compile_segment_edit(
            session_id,
            operations,
            expected_version,
            enhance=_optional_bool(payload, "enhance"),
        )
        settings = _settings(payload)
        model_id = str(
            payload.get("model_id")
            or state.selected_model_id
            or service.default_model_id
        )
        service.get_model_spec(model_id)
        return EditRequest(
            source_audio_path=source.audio_path,
            source_text=source_text,
            compiled_edit=compiled,
            source_revision_id=source.revision_id,
            model_id=model_id,
            use_xvector=_edit_xvector_mode(payload),
            xvector_allowed=source.use_xvector,
            settings=settings,
        )

    @app.api(
        name="synthesize_tts",
        api_visibility="undocumented",
        concurrency_id="dots-tts-model",
        concurrency_limit=1,
    )
    def synthesize_tts(session_id: str, text: str, settings_json: str) -> Iterator[str]:
        yield json.dumps({"phase": "preparing", "progress": 0.1})
        try:
            payload = json.loads(settings_json or "{}")
            request = prepare_tts_request(session_id, text, payload)
            yield json.dumps({"phase": "inference", "progress": None})
            result = service.generate_tts(session_id, request)
            yield json.dumps({"phase": "saving", "progress": 0.9})
            store.add_result(session_id, result)
            yield json.dumps(
                {
                    "phase": "complete",
                    "progress": 1.0,
                    "snapshot": store.snapshot(session_id),
                }
            )
        except Exception as exc:
            yield json.dumps(
                {"phase": "error", "progress": None, "message": str(exc)}
            )

    @app.api(
        name="synthesize_edit",
        api_visibility="undocumented",
        concurrency_id="dots-tts-model",
        concurrency_limit=1,
    )
    def synthesize_edit(
        session_id: str,
        operations_json: str,
        expected_version: str,
        settings_json: str,
    ) -> Iterator[str]:
        yield json.dumps({"phase": "preparing", "progress": 0.1})
        try:
            payload = json.loads(settings_json or "{}")
            request = prepare_edit_request(
                session_id,
                list(json.loads(operations_json or "[]")),
                expected_version,
                payload,
            )
            yield json.dumps({"phase": "inference", "progress": None})
            result = service.generate_edit(session_id, request)
            yield json.dumps({"phase": "saving", "progress": 0.9})
            store.add_result(session_id, result)
            yield json.dumps(
                {
                    "phase": "complete",
                    "progress": 1.0,
                    "snapshot": store.snapshot(session_id),
                }
            )
        except Exception as exc:
            yield json.dumps(
                {"phase": "error", "progress": None, "message": str(exc)}
            )

    if uploads_enabled and recognition_jobs is not None:
        if gpu_recognition_handler is None:
            raise ValueError(
                "gpu_recognition_handler is required when recognition is enabled."
            )

        @app.post("/api/session/{session_id}/recognition/prepare")
        def prepare_recognition(
            session_id: str,
            audio: UploadFile,
        ) -> dict[str, str]:
            try:
                return recognition_jobs.create(session_id, audio)
            except ValueError as exc:
                raise fail(exc) from exc

        @app.delete("/api/session/{session_id}/recognition/{job_id}")
        def cancel_recognition(
            session_id: str,
            job_id: str,
        ) -> Response:
            try:
                recognition_jobs.cancel(session_id, job_id)
                return Response(status_code=204)
            except ValueError as exc:
                raise fail(exc) from exc

        app.api(
            name="recognize",
            api_visibility="undocumented",
            concurrency_id="dots-tts-model",
            concurrency_limit=1,
        )(gpu_recognition_handler)

    return app, store
