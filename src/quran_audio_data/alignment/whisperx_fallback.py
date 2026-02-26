from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from rapidfuzz import fuzz

from quran_audio_data.alignment.base import AlignmentOutput, AlignmentError, EngineUnavailable
from quran_audio_data.schema import AyahTiming, WordTiming
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
        ayahs = _derive_ayahs(mapped_words, source="fallback")

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
            start = _to_float(word.get("start"))
            end = _to_float(word.get("end"))
            if not text or start is None or end is None:
                continue

            out.append(
                WhisperWord(
                    text_norm=normalize_arabic(text),
                    start_s=start,
                    end_s=end,
                    confidence=_to_float(word.get("score")),
                )
            )

    return sorted(out, key=lambda x: x.start_s)


def _map_words(
    *,
    canonical_words: list[CanonicalWord],
    predicted_words: list[WhisperWord],
    audio_duration_s: float,
) -> list[WordTiming]:
    mapped: list[WordTiming] = []
    cursor = 0
    matched_idx: dict[int, int] = {}

    for i, canon in enumerate(canonical_words):
        best_j: int | None = None
        best_score = -1.0

        search_end = min(len(predicted_words), cursor + 16)
        for j in range(cursor, search_end):
            candidate = predicted_words[j]
            score = fuzz.ratio(canon.text_norm, candidate.text_norm)
            if score > best_score:
                best_score = score
                best_j = j
            if score >= 98:
                break

        if best_j is not None and best_score >= 45:
            matched_idx[i] = best_j
            cursor = best_j + 1

    for i, canon in enumerate(canonical_words):
        pred_index = matched_idx.get(i)
        if pred_index is not None:
            pred = predicted_words[pred_index]
            start_s, end_s = pred.start_s, pred.end_s
            confidence = pred.confidence
        else:
            start_s, end_s = _interpolate_slot(
                index=i,
                total=len(canonical_words),
                matched_idx=matched_idx,
                predicted_words=predicted_words,
                audio_duration_s=audio_duration_s,
            )
            confidence = None

        mapped.append(
            WordTiming(
                surah=canon.surah,
                ayah=canon.ayah,
                word_index_global=canon.word_index_global,
                word_index_in_ayah=canon.word_index_in_ayah,
                text_uthmani=canon.text_uthmani,
                text_norm=canon.text_norm,
                start_s=start_s,
                end_s=end_s,
                confidence=confidence,
            )
        )

    return mapped


def _interpolate_slot(
    *,
    index: int,
    total: int,
    matched_idx: dict[int, int],
    predicted_words: list[WhisperWord],
    audio_duration_s: float,
) -> tuple[float, float]:
    prev_i = max((i for i in matched_idx if i < index), default=None)
    next_i = min((i for i in matched_idx if i > index), default=None)

    if prev_i is not None:
        prev_word = predicted_words[matched_idx[prev_i]]
        left = prev_word.end_s
    else:
        left = 0.0

    if next_i is not None:
        next_word = predicted_words[matched_idx[next_i]]
        right = next_word.start_s
    else:
        right = max(audio_duration_s, predicted_words[-1].end_s)

    gap_count = (next_i - prev_i - 1) if prev_i is not None and next_i is not None else total
    gap_count = max(gap_count, 1)

    if prev_i is not None and next_i is not None:
        rank = index - prev_i - 1
    elif prev_i is None and next_i is not None:
        rank = index
    else:
        rank = max(0, index - (prev_i if prev_i is not None else 0))

    width = max(0.0, right - left)
    slot = width / gap_count
    start_s = left + (rank * slot)
    end_s = left + ((rank + 1) * slot)
    return start_s, end_s


def _derive_ayahs(words: list[WordTiming], *, source: str) -> list[AyahTiming]:
    by_ayah: dict[tuple[int, int], list[WordTiming]] = defaultdict(list)
    for word in words:
        by_ayah[(word.surah, word.ayah)].append(word)

    ayahs: list[AyahTiming] = []
    for (surah, ayah), group in sorted(by_ayah.items(), key=lambda item: (item[0][0], item[0][1])):
        start_s = min(item.start_s for item in group)
        end_s = max(item.end_s for item in group)
        ayahs.append(
            AyahTiming(
                surah=surah,
                ayah=ayah,
                start_s=start_s,
                end_s=end_s,
                source=source,
            )
        )
    return ayahs


def _to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
