from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import math
from typing import Any

import orjson
import soundfile as sf
from rapidfuzz import fuzz

from quran_audio_data.text import normalize_arabic


@dataclass(slots=True)
class PredictedWord:
    text_norm: str
    start_s: float
    end_s: float
    confidence: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run NeMo ASR timestamp extraction and map to reference transcript words."
    )
    parser.add_argument("--audio", required=True, help="Path to WAV audio (16k mono recommended)")
    parser.add_argument("--transcript", required=True, help="Path to reference transcript text file")
    parser.add_argument("--model", required=True, help="NeMo pretrained model name")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out", required=True, help="Output JSON path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    audio_path = Path(args.audio)
    transcript_path = Path(args.transcript)
    output_path = Path(args.out)

    transcript_words = load_transcript_words(transcript_path)
    if not transcript_words:
        raise RuntimeError("Transcript is empty; cannot perform alignment")

    audio_duration_s = probe_audio_duration(audio_path)
    model, resolved_device = load_nemo_model(model_name=args.model, device=args.device)
    hypothesis = transcribe_hypothesis(model=model, audio_path=audio_path)

    predicted_words = extract_predicted_words(
        hypothesis=hypothesis,
        transcript_words=transcript_words,
        audio_duration_s=audio_duration_s,
    )
    mapped_words = map_reference_words(
        transcript_words=transcript_words,
        predicted_words=predicted_words,
        audio_duration_s=audio_duration_s,
    )

    output = {
        "engine": "nemo-runner",
        "model": args.model,
        "device": resolved_device,
        "words": mapped_words,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(output, option=orjson.OPT_INDENT_2))
    return 0


def load_transcript_words(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8").strip()
    return [piece for piece in text.split() if piece]


def probe_audio_duration(path: Path) -> float:
    info = sf.info(str(path))
    return float(info.duration)


def load_nemo_model(*, model_name: str, device: str):
    try:
        import torch
        from nemo.collections.asr.models import ASRModel
        from nemo.collections.asr.models.hybrid_rnnt_ctc_models import EncDecHybridRNNTCTCModel
    except Exception as exc:
        raise RuntimeError(
            "NeMo dependencies are unavailable. Install with: uv sync --extra cpu "
            "(or --extra gpu with proper PyTorch CUDA wheel)."
        ) from exc

    if device == "auto":
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        resolved_device = device

    map_location = torch.device(resolved_device)
    model = ASRModel.from_pretrained(model_name=model_name, map_location=map_location)

    if isinstance(model, EncDecHybridRNNTCTCModel):
        try:
            model.change_decoding_strategy(decoder_type="ctc")
        except Exception:
            pass

    configure_timestamps_decoding(model)

    model = model.eval()
    try:
        model = model.to(map_location)
    except Exception:
        pass

    return model, resolved_device


def configure_timestamps_decoding(model: Any) -> None:
    decoding_cfg = None

    if hasattr(model, "cfg") and hasattr(model.cfg, "ctc_decoding"):
        decoding_cfg = model.cfg.ctc_decoding
    elif hasattr(model, "_cfg") and hasattr(model._cfg, "ctc_decoding"):
        decoding_cfg = model._cfg.ctc_decoding

    if decoding_cfg is None:
        return

    # Best-effort config mutation across Nemo/OmegaConf variants.
    for key, value in {
        "compute_timestamps": True,
        "preserve_alignments": True,
        "word_seperator": " ",
        "segment_seperators": [".", "?", "!", "..."],
        "ctc_timestamp_type": "word",
    }.items():
        try:
            setattr(decoding_cfg, key, value)
        except Exception:
            pass

    if hasattr(model, "change_decoding_strategy"):
        try:
            model.change_decoding_strategy(decoding_cfg, decoder_type="ctc")
            return
        except Exception:
            pass
        try:
            model.change_decoding_strategy(decoding_cfg)
        except Exception:
            pass


def transcribe_hypothesis(*, model: Any, audio_path: Path):
    attempts = [
        lambda: model.transcribe(
            [str(audio_path)],
            batch_size=1,
            return_hypotheses=True,
            timestamps=True,
            verbose=False,
        ),
        lambda: model.transcribe(
            paths2audio_files=[str(audio_path)],
            batch_size=1,
            return_hypotheses=True,
            timestamps=True,
            verbose=False,
        ),
        lambda: model.transcribe(
            [str(audio_path)],
            batch_size=1,
            return_hypotheses=True,
            verbose=False,
        ),
        lambda: model.transcribe(
            paths2audio_files=[str(audio_path)],
            batch_size=1,
            return_hypotheses=True,
            verbose=False,
        ),
    ]

    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            result = attempt()
            first = first_hypothesis(result)
            if first is not None:
                return first
        except Exception as exc:
            last_exc = exc

    if last_exc is not None:
        raise RuntimeError(f"NeMo transcription failed: {last_exc}") from last_exc
    raise RuntimeError("NeMo transcription failed with no hypothesis output")


def first_hypothesis(result: Any) -> Any | None:
    if result is None:
        return None

    if isinstance(result, list):
        if not result:
            return None
        first = result[0]
        if isinstance(first, list):
            return first[0] if first else None
        return first

    return result


def extract_predicted_words(
    *,
    hypothesis: Any,
    transcript_words: list[str],
    audio_duration_s: float,
) -> list[PredictedWord]:
    timestamps = getattr(hypothesis, "timestamp", None)
    entries: list[dict[str, Any]] = []

    if isinstance(timestamps, dict):
        for key in ("word", "words", "word_timestamps"):
            value = timestamps.get(key)
            if isinstance(value, list) and value:
                entries = [item for item in value if isinstance(item, dict)]
                break

    # Fallback if model exposes direct words list.
    if not entries:
        words_obj = getattr(hypothesis, "words", None)
        if isinstance(words_obj, list) and words_obj and isinstance(words_obj[0], dict):
            entries = [item for item in words_obj if isinstance(item, dict)]

    predicted: list[PredictedWord] = []
    for item in entries:
        text = str(item.get("word") or item.get("text") or "").strip()
        if not text:
            continue

        start = to_float(
            item.get("start")
            or item.get("start_time")
            or item.get("start_s")
            or item.get("timestamp_from")
            or item.get("start_offset")
        )
        end = to_float(
            item.get("end")
            or item.get("end_time")
            or item.get("end_s")
            or item.get("timestamp_to")
            or item.get("end_offset")
        )

        duration = to_float(item.get("duration"))
        if start is not None and end is None and duration is not None:
            end = start + duration

        if start is None or end is None:
            continue

        predicted.append(
            PredictedWord(
                text_norm=normalize_arabic(text),
                start_s=start,
                end_s=end,
                confidence=to_float(item.get("confidence") or item.get("score") or item.get("probability")),
            )
        )

    if not predicted:
        # Keep pipeline alive by falling back to uniform timing over transcript.
        return []

    predicted.sort(key=lambda x: x.start_s)
    predicted = maybe_convert_offsets_to_seconds(predicted, audio_duration_s)
    return predicted


def maybe_convert_offsets_to_seconds(
    predicted_words: list[PredictedWord], audio_duration_s: float
) -> list[PredictedWord]:
    if not predicted_words or audio_duration_s <= 0:
        return predicted_words

    max_end = max(word.end_s for word in predicted_words)
    if max_end <= 0:
        return predicted_words

    # If values are clearly frame/offset indices, scale to audio seconds.
    if max_end > audio_duration_s * 2.0:
        scale = audio_duration_s / max_end
        scaled: list[PredictedWord] = []
        for item in predicted_words:
            scaled.append(
                PredictedWord(
                    text_norm=item.text_norm,
                    start_s=item.start_s * scale,
                    end_s=item.end_s * scale,
                    confidence=item.confidence,
                )
            )
        return scaled

    return predicted_words


def map_reference_words(
    *,
    transcript_words: list[str],
    predicted_words: list[PredictedWord],
    audio_duration_s: float,
) -> list[dict[str, Any]]:
    transcript_norm = [normalize_arabic(word) for word in transcript_words]

    if not predicted_words:
        return build_uniform_words(transcript_words=transcript_words, audio_duration_s=audio_duration_s)

    matched: dict[int, int] = {}
    cursor = 0

    for i, ref in enumerate(transcript_norm):
        best_j: int | None = None
        best_score = -1.0

        search_end = min(len(predicted_words), cursor + 16)
        for j in range(cursor, search_end):
            pred = predicted_words[j]
            score = fuzz.ratio(ref, pred.text_norm)
            if score > best_score:
                best_score = score
                best_j = j
            if score >= 98:
                break

        if best_j is not None and best_score >= 45:
            matched[i] = best_j
            cursor = best_j + 1

    output: list[dict[str, Any]] = []
    for i, word in enumerate(transcript_words):
        j = matched.get(i)
        if j is not None:
            pred = predicted_words[j]
            start_s = pred.start_s
            end_s = pred.end_s
            confidence = pred.confidence
        else:
            start_s, end_s = interpolate_slot(
                index=i,
                total=len(transcript_words),
                matched_idx=matched,
                predicted_words=predicted_words,
                audio_duration_s=audio_duration_s,
            )
            confidence = None

        output.append(
            {
                "text": word,
                "start": max(0.0, float(start_s)),
                "end": max(0.0, float(end_s)),
                "confidence": confidence,
            }
        )

    return output


def build_uniform_words(*, transcript_words: list[str], audio_duration_s: float) -> list[dict[str, Any]]:
    total = max(1, len(transcript_words))
    slot = audio_duration_s / total if audio_duration_s > 0 else 0.0
    rows: list[dict[str, Any]] = []
    for idx, word in enumerate(transcript_words):
        rows.append(
            {
                "text": word,
                "start": idx * slot,
                "end": (idx + 1) * slot,
                "confidence": None,
            }
        )
    return rows


def interpolate_slot(
    *,
    index: int,
    total: int,
    matched_idx: dict[int, int],
    predicted_words: list[PredictedWord],
    audio_duration_s: float,
) -> tuple[float, float]:
    prev_i = max((i for i in matched_idx if i < index), default=None)
    next_i = min((i for i in matched_idx if i > index), default=None)

    if prev_i is not None:
        left = predicted_words[matched_idx[prev_i]].end_s
    else:
        left = 0.0

    if next_i is not None:
        right = predicted_words[matched_idx[next_i]].start_s
    else:
        right = max(audio_duration_s, predicted_words[-1].end_s)

    if prev_i is not None and next_i is not None:
        gap_count = max(1, next_i - prev_i - 1)
        rank = index - prev_i - 1
    elif prev_i is None and next_i is not None:
        gap_count = max(1, next_i)
        rank = index
    else:
        gap_count = max(1, total - (prev_i + 1 if prev_i is not None else 0))
        rank = index - (prev_i + 1 if prev_i is not None else 0)

    width = max(0.0, right - left)
    slot = width / gap_count
    start_s = left + (rank * slot)
    end_s = left + ((rank + 1) * slot)
    if end_s < start_s:
        end_s = start_s
    return start_s, end_s


def to_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


if __name__ == "__main__":
    raise SystemExit(main())
