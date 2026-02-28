from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from quran_audio_data.alignment.base import AlignmentOutput, AlignmentError, EngineUnavailable
from quran_audio_data.alignment.mapping import (
    MappingConfig,
    PredictionSpan,
    derive_ayahs_from_words,
    interpolate_slot,
    map_canonical_words,
    to_prediction_spans,
)
from quran_audio_data.core.parsing import to_float
from quran_audio_data.schema import WordTiming
from quran_audio_data.text.quran_text import CanonicalWord, normalize_arabic


@dataclass(slots=True)
class WhisperWord:
    text_norm: str
    start_s: float
    end_s: float
    confidence: float | None


class WhisperXFallbackAligner:
    def __init__(
        self,
        *,
        model_name: str = "large-v3",
        align_model_name: str = "jonatasgrosman/wav2vec2-large-xlsr-53-arabic",
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.align_model_name = align_model_name
        self.batch_size = batch_size

    def align(
        self,
        *,
        audio_wav_path: str,
        canonical_words: list[CanonicalWord],
        audio_duration_s: float,
        device: str,
    ) -> AlignmentOutput:
        whisperx = _import_whisperx()
        resolved_device = _resolve_device(device)

        try:
            audio = whisperx.load_audio(audio_wav_path)
            model = whisperx.load_model(
                self.model_name,
                resolved_device,
                language="ar",
                compute_type="float16" if resolved_device == "cuda" else "int8",
            )
            result = model.transcribe(audio, batch_size=self.batch_size)
            language_code = result.get("language", "ar")

            try:
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=resolved_device,
                    model_name=self.align_model_name,
                )
            except TypeError:
                model_a, metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=resolved_device,
                )

            aligned = whisperx.align(
                result.get("segments", []),
                model_a,
                metadata,
                audio,
                resolved_device,
                return_char_alignments=False,
            )
        except Exception as exc:
            raise AlignmentError(f"WhisperX alignment failed: {exc}") from exc

        predicted_words = _extract_predicted_words(aligned)
        if not predicted_words:
            raise AlignmentError("WhisperX produced no word timestamps")

        mapped_words = _map_words(
            canonical_words=canonical_words,
            predicted_words=predicted_words,
            audio_duration_s=audio_duration_s,
        )
        ayahs = derive_ayahs_from_words(words=mapped_words, source="fallback")

        return AlignmentOutput(
            ayahs=ayahs,
            words=mapped_words,
            engine_name="whisperx",
            engine_model=self.model_name,
            device=resolved_device,
            source="fallback",
        )


def _import_whisperx() -> Any:
    try:
        try:
            import torchaudio  # type: ignore

            if not hasattr(torchaudio, "AudioMetaData"):
                backend_common = getattr(getattr(torchaudio, "backend", None), "common", None)
                if backend_common is not None and hasattr(backend_common, "AudioMetaData"):
                    torchaudio.AudioMetaData = backend_common.AudioMetaData  # type: ignore[attr-defined]
        except Exception:
            pass

        import whisperx  # type: ignore

        return whisperx
    except Exception as exc:
        raise EngineUnavailable(
            "whisperx is not available. Install optional deps: uv sync --extra cpu (or --extra gpu)."
        ) from exc


def _resolve_device(device: str) -> str:
    if device in {"cpu", "cuda"}:
        return device

    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _extract_predicted_words(aligned_payload: dict[str, Any]) -> list[WhisperWord]:
    out: list[WhisperWord] = []
    segments = aligned_payload.get("segments")
    if not isinstance(segments, list):
        return out

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        words = segment.get("words")
        if not isinstance(words, list):
            continue
        for word in words:
            if not isinstance(word, dict):
                continue
            text = str(word.get("word", "")).strip()
            start = to_float(word.get("start"))
            end = to_float(word.get("end"))
            if not text or start is None or end is None:
                continue

            out.append(
                WhisperWord(
                    text_norm=normalize_arabic(text),
                    start_s=start,
                    end_s=end,
                    confidence=to_float(word.get("score")),
                )
            )

    return sorted(out, key=lambda x: x.start_s)


def _map_words(
    *,
    canonical_words: list[CanonicalWord],
    predicted_words: list[WhisperWord],
    audio_duration_s: float,
) -> list[WordTiming]:
    spans = to_prediction_spans(
        predicted_words=predicted_words,
        text_getter=lambda item: item.text_norm,
        start_getter=lambda item: item.start_s,
        end_getter=lambda item: item.end_s,
        confidence_getter=lambda item: item.confidence,
    )
    return map_canonical_words(
        canonical_words=canonical_words,
        predicted_words=spans,
        audio_duration_s=audio_duration_s,
        config=MappingConfig(
            engine_candidate="whisperx",
            search_window=16,
            exact_break_score=98.0,
            min_match_score=45.0,
            matched_origin="native",
            unmatched_origin="interpolated",
        ),
    )


def _interpolate_slot(
    *,
    index: int,
    total: int,
    matched_idx: dict[int, int],
    predicted_words: list[WhisperWord],
    audio_duration_s: float,
) -> tuple[float, float]:
    spans = [
        PredictionSpan(
            prediction=item,
            text_norm=item.text_norm,
            start_s=item.start_s,
            end_s=item.end_s,
            confidence=item.confidence,
        )
        for item in predicted_words
    ]
    return interpolate_slot(
        index=index,
        total=total,
        matched_idx=matched_idx,
        predicted_words=spans,
        audio_duration_s=audio_duration_s,
    )
