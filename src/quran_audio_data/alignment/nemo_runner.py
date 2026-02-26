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
        description="Run NeMo CTC forced alignment and emit word timestamps."
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

    words, resolved_device = forced_align(
        audio_path=audio_path,
        transcript_words=transcript_words,
        model_name=args.model,
        device=args.device,
    )

    output = {
        "engine": "nemo-forced-align",
        "model": args.model,
        "device": resolved_device,
        "words": words,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(orjson.dumps(output, option=orjson.OPT_INDENT_2))
    return 0


def forced_align(
    *,
    audio_path: Path,
    transcript_words: list[str],
    model_name: str,
    device: str,
) -> tuple[list[dict[str, Any]], str]:
    audio_duration_s = probe_audio_duration(audio_path)

    model, torch, resolved_device = load_nemo_model(model_name=model_name, device=device)

    try:
        from nemo.collections.asr.parts.utils.aligner_utils import (
            add_t_start_end_to_utt_obj,
            get_batch_variables,
            viterbi_decoding,
        )
    except Exception as exc:
        raise RuntimeError(
            "NeMo forced-align utilities are unavailable. Install NeMo with ASR extras."
        ) from exc

    transcript_text = " ".join(transcript_words)
    (
        log_probs_batch,
        y_batch,
        t_batch,
        u_batch,
        utt_obj_batch,
        output_timestep_duration,
    ) = get_batch_variables(
        audio=[str(audio_path)],
        model=model,
        segment_separators=None,
        align_using_pred_text=False,
        audio_filepath_parts_in_utt_id=1,
        gt_text_batch=[transcript_text],
        output_timestep_duration=None,
        simulate_cache_aware_streaming=False,
        use_buffered_chunked_streaming=False,
        buffered_chunk_params={},
    )

    alignments = viterbi_decoding(
        log_probs_batch,
        y_batch,
        t_batch,
        u_batch,
        torch.device(resolved_device),
    )

    utt = add_t_start_end_to_utt_obj(
        utt_obj_batch[0],
        alignments[0],
        output_timestep_duration,
    )

    predicted_words = extract_predicted_words_from_utt(utt)
    mapped_words = map_reference_words(
        transcript_words=transcript_words,
        predicted_words=predicted_words,
        audio_duration_s=audio_duration_s,
    )

    return mapped_words, resolved_device


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

    model = model.eval()
    try:
        model = model.to(map_location)
    except Exception:
        pass

    return model, torch, resolved_device


def extract_predicted_words_from_utt(utt_obj: Any) -> list[PredictedWord]:
    predicted: list[PredictedWord] = []

    # `segments_and_tokens` is a mixed list; words live under segment.words_and_tokens.
    for segment_or_token in getattr(utt_obj, "segments_and_tokens", []):
        words_and_tokens = getattr(segment_or_token, "words_and_tokens", None)
        if not isinstance(words_and_tokens, list):
            continue

        for maybe_word in words_and_tokens:
            if not hasattr(maybe_word, "tokens"):
                continue

            text = str(getattr(maybe_word, "text", "") or "").strip()
            start = to_float(getattr(maybe_word, "t_start", None))
            end = to_float(getattr(maybe_word, "t_end", None))
            if not text or start is None or end is None:
                continue

            predicted.append(
                PredictedWord(
                    text_norm=normalize_arabic(text),
                    start_s=start,
                    end_s=end,
                    confidence=None,
                )
            )

    predicted.sort(key=lambda item: item.start_s)
    return predicted


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
    matched_scores: dict[int, float] = {}
    cursor = 0

    for i, ref in enumerate(transcript_norm):
        best_j: int | None = None
        best_score = -1.0

        search_end = min(len(predicted_words), cursor + 24)
        for j in range(cursor, search_end):
            pred = predicted_words[j]
            score = fuzz.ratio(ref, pred.text_norm)
            if score > best_score:
                best_score = score
                best_j = j
            if score >= 99:
                break

        if best_j is not None and best_score >= 55:
            matched[i] = best_j
            matched_scores[i] = float(best_score)
            cursor = best_j + 1

    output: list[dict[str, Any]] = []
    for i, word in enumerate(transcript_words):
        j = matched.get(i)
        if j is not None:
            pred = predicted_words[j]
            start_s = pred.start_s
            end_s = pred.end_s
            confidence = pred.confidence
            origin = "native"
            match_score = matched_scores.get(i)
        else:
            start_s, end_s = interpolate_slot(
                index=i,
                total=len(transcript_words),
                matched_idx=matched,
                predicted_words=predicted_words,
                audio_duration_s=audio_duration_s,
            )
            confidence = None
            origin = "interpolated"
            match_score = None

        output.append(
            {
                "text": word,
                "start": max(0.0, float(start_s)),
                "end": max(0.0, float(end_s)),
                "confidence": confidence,
                "alignment_origin": origin,
                "match_score": match_score,
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
                "alignment_origin": "distributed",
                "match_score": None,
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
