from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
from tempfile import TemporaryDirectory
from typing import Any

import orjson
import soundfile as sf

from quran_audio_data.alignment.mapping import (
    MappingConfig,
    PredictionSpan,
    interpolate_slot as shared_interpolate_slot,
    map_canonical_words,
    to_prediction_spans,
)
from quran_audio_data.core.parsing import to_float
from quran_audio_data.text import CanonicalWord
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
    parser.add_argument(
        "--transcript", required=True, help="Path to reference transcript text file"
    )
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
    enable_chunked_logits = bool(resolved_device == "cpu" and audio_duration_s >= 15 * 60)
    hypotheses: list[Any] | None = None
    if enable_chunked_logits:
        logits = _transcribe_log_probs_chunked(
            audio_path=audio_path,
            model=model,
            torch=torch,
            chunk_s=30.0,
        )
        hypotheses = [_InlineHypothesis(y_sequence=logits, text="")]

    (
        log_probs_batch,
        y_batch,
        t_batch,
        u_batch,
        utt_obj_batch,
        output_timestep_duration,
    ) = get_batch_variables(
        audio=hypotheses if hypotheses is not None else [str(audio_path)],
        model=model,
        segment_separators=None,
        align_using_pred_text=False,
        audio_filepath_parts_in_utt_id=1,
        gt_text_batch=[transcript_text],
        output_timestep_duration=None,
        simulate_cache_aware_streaming=False,
        use_buffered_chunked_streaming=False,
        buffered_chunk_params={},
        has_hypotheses=hypotheses is not None,
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


@dataclass(slots=True)
class _InlineHypothesis:
    y_sequence: Any
    text: str


def _transcribe_log_probs_chunked(
    *,
    audio_path: Path,
    model: Any,
    torch: Any,
    chunk_s: float,
) -> Any:
    if chunk_s <= 0:
        raise ValueError(f"chunk_s must be > 0, got: {chunk_s}")

    info = sf.info(str(audio_path))
    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    if channels != 1:
        raise RuntimeError(f"expected mono audio for NeMo chunking, got channels={channels}")

    frames_per_chunk = int(round(chunk_s * sample_rate))
    frames_per_chunk = max(1, frames_per_chunk)

    chunk_logits: list[tuple[Path, int]] = []
    with TemporaryDirectory(prefix="qad_nemo_chunks_") as tmp:
        tmp_dir = Path(tmp)
        with sf.SoundFile(str(audio_path), mode="r") as reader:
            chunk_idx = 0
            while True:
                audio = reader.read(frames_per_chunk, dtype="float32")
                if audio is None or len(audio) == 0:
                    break

                chunk_wav = tmp_dir / f"chunk_{chunk_idx:05d}.wav"
                sf.write(str(chunk_wav), audio, sample_rate)

                with torch.no_grad():
                    hypotheses = model.transcribe(
                        [str(chunk_wav)],
                        return_hypotheses=True,
                        batch_size=1,
                        verbose=False,
                    )

                if isinstance(hypotheses, tuple) and len(hypotheses) == 2:
                    hypotheses = hypotheses[0]
                if not hypotheses:
                    raise RuntimeError(f"empty hypotheses for chunk {chunk_idx}")

                hyp = hypotheses[0]
                logits = getattr(hyp, "y_sequence", None)
                if logits is None or not hasattr(logits, "shape"):
                    raise RuntimeError(f"missing y_sequence on hypothesis for chunk {chunk_idx}")

                logits = logits.detach().cpu()
                frames = int(logits.shape[0])
                if frames <= 0:
                    raise RuntimeError(f"empty logits for chunk {chunk_idx}")

                logits_path = tmp_dir / f"chunk_{chunk_idx:05d}.pt"
                torch.save(logits, str(logits_path))
                chunk_logits.append((logits_path, frames))
                chunk_idx += 1

        if not chunk_logits:
            raise RuntimeError("no chunk logits produced")

        total_frames = sum(frames for _, frames in chunk_logits)
        first = torch.load(str(chunk_logits[0][0]), map_location="cpu")
        vocab_dim = int(first.shape[1])
        full = torch.empty((total_frames, vocab_dim), dtype=first.dtype)

        offset = 0
        for path, frames in chunk_logits:
            piece = torch.load(str(path), map_location="cpu")
            if int(piece.shape[1]) != vocab_dim:
                raise RuntimeError("vocab dim mismatch while stitching chunk logits")
            full[offset : offset + frames] = piece
            offset += frames

        return full


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
    if not predicted_words:
        return build_uniform_words(
            transcript_words=transcript_words, audio_duration_s=audio_duration_s
        )

    canonical_words = [
        CanonicalWord(
            surah=1,
            ayah=1,
            word_index_global=index,
            word_index_in_ayah=index,
            text_uthmani=word,
            text_norm=normalize_arabic(word),
        )
        for index, word in enumerate(transcript_words, start=1)
    ]
    prediction_spans = to_prediction_spans(
        predicted_words=predicted_words,
        text_getter=lambda item: item.text_norm,
        start_getter=lambda item: item.start_s,
        end_getter=lambda item: item.end_s,
        confidence_getter=lambda item: item.confidence,
    )
    mapped = map_canonical_words(
        canonical_words=canonical_words,
        predicted_words=prediction_spans,
        audio_duration_s=audio_duration_s,
        config=MappingConfig(
            engine_candidate="nemo",
            search_window=24,
            exact_break_score=99.0,
            min_match_score=55.0,
            matched_origin="native",
            unmatched_origin="interpolated",
        ),
    )
    return [
        {
            "text": item.text_uthmani,
            "start": max(0.0, float(item.start_s)),
            "end": max(0.0, float(item.end_s)),
            "confidence": item.confidence,
            "alignment_origin": item.alignment_origin,
            "match_score": item.match_score,
        }
        for item in mapped
    ]


def build_uniform_words(
    *, transcript_words: list[str], audio_duration_s: float
) -> list[dict[str, Any]]:
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
    return shared_interpolate_slot(
        index=index,
        total=total,
        matched_idx=matched_idx,
        predicted_words=spans,
        audio_duration_s=audio_duration_s,
    )


if __name__ == "__main__":
    raise SystemExit(main())
