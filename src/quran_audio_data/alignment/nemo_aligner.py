from __future__ import annotations

from pathlib import Path
import os
import shlex
import subprocess
import sys
import tempfile
from typing import Any

import orjson
from rapidfuzz import fuzz

from quran_audio_data.alignment.base import AlignmentOutput, AlignmentError
from quran_audio_data.alignment.mapping import derive_ayahs_from_words
from quran_audio_data.core.parsing import safe_get, to_float
from quran_audio_data.schema import WordTiming
from quran_audio_data.text.quran_text import CanonicalWord, normalize_arabic


class NemoAligner:
    """Wrapper around NeMo alignment runner via configurable command invocation.

    By default this class calls the built-in `quran_audio_data.alignment.nemo_runner`
    module. `QAD_NEMO_ALIGN_CMD` can override this behavior if needed.

    Available placeholders:
    - {audio_wav}
    - {transcript_txt}
    - {output_json}
    - {model}
    - {device}
    """

    def __init__(
        self,
        *,
        model_name: str = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
        command_template: str | None = None,
    ) -> None:
        self.model_name = model_name
        default_runner_cmd = (
            f"{shlex.quote(sys.executable)} -m quran_audio_data.alignment.nemo_runner "
            "--audio {audio_wav} --transcript {transcript_txt} --model {model} "
            "--device {device} --out {output_json}"
        )
        self.command_template = (
            command_template or os.getenv("QAD_NEMO_ALIGN_CMD") or default_runner_cmd
        )

    def align(
        self,
        *,
        audio_wav_path: str,
        canonical_words: list[CanonicalWord],
        audio_duration_s: float,
        device: str,
    ) -> AlignmentOutput:
        if not canonical_words:
            raise AlignmentError("No canonical words available for alignment")

        resolved_device = _resolve_device(device)

        with tempfile.TemporaryDirectory(prefix="qad_nemo_") as temp_dir:
            temp_root = Path(temp_dir)
            transcript_txt = temp_root / "transcript.txt"
            output_json = temp_root / "output.json"

            transcript_txt.write_text(
                " ".join(word.text_norm for word in canonical_words),
                encoding="utf-8",
            )

            cmd = self.command_template.format(
                audio_wav=shlex.quote(str(audio_wav_path)),
                transcript_txt=shlex.quote(str(transcript_txt)),
                output_json=shlex.quote(str(output_json)),
                model=shlex.quote(self.model_name),
                device=shlex.quote(resolved_device),
            )

            proc = subprocess.run(
                cmd,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if proc.returncode != 0:
                raise AlignmentError(
                    "NeMo command failed with non-zero exit code "
                    f"{proc.returncode}: {proc.stderr.strip()}"
                )

            if not output_json.exists():
                raise AlignmentError(
                    f"NeMo command completed but did not produce output file: {output_json}"
                )

            payload = orjson.loads(output_json.read_bytes())
            words = _normalize_nemo_output(
                payload=payload,
                canonical_words=canonical_words,
                audio_duration_s=audio_duration_s,
            )

        ayahs = derive_ayahs_from_words(words=words, source="aligned")
        return AlignmentOutput(
            ayahs=ayahs,
            words=words,
            engine_name="nemo",
            engine_model=self.model_name,
            device=resolved_device,
            source="aligned",
        )


def _normalize_nemo_output(
    *,
    payload: Any,
    canonical_words: list[CanonicalWord],
    audio_duration_s: float,
) -> list[WordTiming]:
    raw_words: list[dict[str, Any]] = []

    if isinstance(payload, dict):
        if isinstance(payload.get("words"), list):
            raw_words = [item for item in payload["words"] if isinstance(item, dict)]
        elif isinstance(payload.get("word_timestamps"), list):
            raw_words = [item for item in payload["word_timestamps"] if isinstance(item, dict)]
    elif isinstance(payload, list):
        raw_words = [item for item in payload if isinstance(item, dict)]

    if not raw_words:
        raise AlignmentError("NeMo output missing word timestamp list")

    mapped: list[WordTiming] = []
    count = len(canonical_words)

    for idx, canon in enumerate(canonical_words):
        sample = raw_words[idx] if idx < len(raw_words) else None
        start = to_float(safe_get(sample, "start", "start_s", "timestamp_from"))
        end = to_float(safe_get(sample, "end", "end_s", "timestamp_to"))
        sample_text_raw = safe_get(sample, "text", "word", "token")
        sample_text_norm = normalize_arabic(str(sample_text_raw)) if sample_text_raw else None
        raw_match_score = to_float(safe_get(sample, "match_score"))
        match_score = (
            raw_match_score
            if raw_match_score is not None
            else (
                float(fuzz.ratio(canon.text_norm, sample_text_norm)) if sample_text_norm else None
            )
        )
        sample_origin = safe_get(sample, "alignment_origin")

        if start is None or end is None:
            start, end = _distributed_slot(audio_duration_s, count, idx)
            origin = "distributed"
        else:
            origin = (
                str(sample_origin)
                if sample_origin in {"native", "interpolated", "distributed"}
                else "native"
            )

        mapped.append(
            WordTiming(
                surah=canon.surah,
                ayah=canon.ayah,
                word_index_global=canon.word_index_global,
                word_index_in_ayah=canon.word_index_in_ayah,
                text_uthmani=canon.text_uthmani,
                text_norm=canon.text_norm,
                start_s=start,
                end_s=end,
                confidence=to_float(safe_get(sample, "confidence", "score")),
                alignment_origin=origin,
                match_score=match_score,
                engine_candidate="nemo",
            )
        )

    return mapped


def _resolve_device(device: str) -> str:
    if device in {"cpu", "cuda"}:
        return device
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _distributed_slot(duration_s: float, count: int, idx: int) -> tuple[float, float]:
    if count <= 0:
        return 0.0, 0.0
    slot = duration_s / count if duration_s > 0 else 0.0
    start = idx * slot
    end = (idx + 1) * slot
    return start, end
