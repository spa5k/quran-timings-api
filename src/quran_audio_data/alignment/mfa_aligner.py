from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
import shlex
import shutil
import subprocess
import tempfile
from typing import Any

import orjson
from rapidfuzz import fuzz

from quran_audio_data.alignment.base import AlignmentOutput, AlignmentError, EngineUnavailable
from quran_audio_data.schema import AyahTiming, WordTiming
from quran_audio_data.text.quran_text import CanonicalWord, normalize_arabic


class MFAAligner:
    """Montreal Forced Aligner wrapper.

    This wrapper creates a temporary one-file corpus and invokes MFA alignment.
    It expects `mfa` to be installed and available in PATH.
    """

    def __init__(
        self,
        *,
        acoustic_model: str = "english_mfa",
        dictionary: str = "__auto_spn__",
        command_template: str | None = None,
        docker_image: str = "mmcauliffe/montreal-forced-aligner:latest",
        mfa_root_dir: str | Path = ".cache/mfa",
    ) -> None:
        self.acoustic_model = acoustic_model
        self.dictionary = dictionary
        self.command_template = command_template or os.getenv("QAD_MFA_ALIGN_CMD")
        self.docker_image = docker_image
        self.mfa_root_dir = Path(mfa_root_dir).resolve()
        self._local_mfa = shutil.which("mfa")
        self._docker = shutil.which("docker")
        self._mode = self._resolve_mode()
        self._availability_error = self._build_availability_error()

    def is_available(self) -> bool:
        return self._mode != "unavailable"

    def availability_error(self) -> str:
        return self._availability_error

    def _resolve_mode(self) -> str:
        if self.command_template:
            return "template"
        if self._local_mfa is not None and _probe_local_mfa(self._local_mfa):
            return "local"
        if self._docker is not None:
            return "docker"
        return "unavailable"

    def _build_availability_error(self) -> str:
        if self._mode != "unavailable":
            return ""
        if self.command_template:
            return ""
        if self._local_mfa is not None and self._mode != "local":
            return (
                "Found `mfa` binary, but it failed to start. "
                "Install MFA via conda or provide a working command via QAD_MFA_ALIGN_CMD."
            )
        if self._docker is None:
            return "Neither a working `mfa` binary nor Docker is available."
        return ""

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
        if self._mode == "unavailable":
            raise EngineUnavailable(self._availability_error or "MFA is unavailable")

        with tempfile.TemporaryDirectory(prefix="qad_mfa_") as temp_dir:
            temp_root = Path(temp_dir)
            corpus_dir = temp_root / "corpus"
            output_dir = temp_root / "out"
            corpus_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

            wav_src = Path(audio_wav_path)
            wav_dst = corpus_dir / "input.wav"
            shutil.copy2(wav_src, wav_dst)

            transcript_txt = corpus_dir / "input.lab"
            transcript_txt.write_text(
                " ".join(word.text_norm for word in canonical_words),
                encoding="utf-8",
            )
            dictionary_arg = self._resolve_dictionary_arg(
                canonical_words=canonical_words,
                corpus_dir=corpus_dir,
            )

            if self._mode == "template":
                cmd = self.command_template.format(
                    corpus_dir=shlex.quote(str(corpus_dir)),
                    output_dir=shlex.quote(str(output_dir)),
                    dictionary=shlex.quote(dictionary_arg),
                    acoustic_model=shlex.quote(self.acoustic_model),
                    audio_wav=shlex.quote(str(wav_dst)),
                    transcript_txt=shlex.quote(str(transcript_txt)),
                )
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            elif self._mode == "local":
                cmd_parts = [
                    self._local_mfa or "mfa",
                    "align",
                    "--clean",
                    "--single_speaker",
                    "--output_format",
                    "json",
                    str(corpus_dir),
                    dictionary_arg,
                    self.acoustic_model,
                    str(output_dir),
                ]
                proc = subprocess.run(
                    cmd_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            elif self._mode == "docker":
                self.mfa_root_dir.mkdir(parents=True, exist_ok=True)
                cmd_parts = [
                    self._docker or "docker",
                    "run",
                    "--rm",
                    "-e",
                    "MFA_ROOT_DIR=/mfa-root",
                    "-v",
                    f"{corpus_dir}:/corpus",
                    "-v",
                    f"{output_dir}:/out",
                    "-v",
                    f"{self.mfa_root_dir}:/mfa-root",
                    self.docker_image,
                    "mfa",
                    "align",
                    "--clean",
                    "--single_speaker",
                    "--output_format",
                    "json",
                    "/corpus",
                    (
                        f"/corpus/{Path(dictionary_arg).name}"
                        if Path(dictionary_arg).exists()
                        else dictionary_arg
                    ),
                    self.acoustic_model,
                    "/out",
                ]
                proc = subprocess.run(
                    cmd_parts,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
            else:
                raise EngineUnavailable(self._availability_error or "MFA is unavailable")

            if proc.returncode != 0:
                raise AlignmentError(
                    "MFA alignment failed with non-zero exit code "
                    f"{proc.returncode}: {proc.stderr.strip()}"
                )

            payload = _load_mfa_output(output_dir)
            predicted_words = _extract_mfa_words(payload)
            if not predicted_words:
                raise AlignmentError("MFA output missing word timestamps")

            words = _map_words(
                canonical_words=canonical_words,
                predicted_words=predicted_words,
                audio_duration_s=audio_duration_s,
            )

        ayahs = _derive_ayahs(words, source="aligned")
        return AlignmentOutput(
            ayahs=ayahs,
            words=words,
            engine_name="mfa",
            engine_model=f"{self.acoustic_model}|{self.dictionary}",
            device=device,
            source="aligned",
        )

    def _resolve_dictionary_arg(
        self,
        *,
        canonical_words: list[CanonicalWord],
        corpus_dir: Path,
    ) -> str:
        if self.dictionary != "__auto_spn__":
            dictionary_path = Path(self.dictionary)
            if dictionary_path.exists():
                copied = corpus_dir / dictionary_path.name
                shutil.copy2(dictionary_path, copied)
                return str(copied)
            return self.dictionary

        dictionary_path = corpus_dir / "auto_spn.dict"
        seen: set[str] = set()
        lines: list[str] = []
        for word in canonical_words:
            token = word.text_norm.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            lines.append(f"{token} spn")
        dictionary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return str(dictionary_path)


def _probe_local_mfa(mfa_bin: str) -> bool:
    try:
        proc = subprocess.run(
            [mfa_bin, "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return False
    return proc.returncode == 0


def _load_mfa_output(output_dir: Path) -> Any:
    json_candidates = sorted(output_dir.rglob("*.json"))
    for file_path in json_candidates:
        try:
            return orjson.loads(file_path.read_bytes())
        except Exception:
            continue
    raise AlignmentError(f"MFA did not produce JSON output under {output_dir}")


def _extract_mfa_words(payload: Any) -> list[tuple[str, float, float]]:
    words: list[tuple[str, float, float]] = []

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            tiers = node.get("tiers")
            if isinstance(tiers, dict):
                words_tier = tiers.get("words")
                if isinstance(words_tier, dict):
                    entries = words_tier.get("entries")
                    if isinstance(entries, list):
                        for entry in entries:
                            if not isinstance(entry, list) or len(entry) < 3:
                                continue
                            begin_f = _to_float(entry[0])
                            end_f = _to_float(entry[1])
                            label = str(entry[2]).strip() if entry[2] is not None else ""
                            if label and label != "<eps>" and begin_f is not None and end_f is not None:
                                words.append((label, begin_f, end_f))

            label = node.get("label") or node.get("text") or node.get("word")
            begin = node.get("begin") or node.get("start") or node.get("xmin")
            end = node.get("end") or node.get("stop") or node.get("xmax")

            if isinstance(label, str) and label.strip() and label.strip() != "<eps>":
                begin_f = _to_float(begin)
                end_f = _to_float(end)
                if begin_f is not None and end_f is not None:
                    words.append((label.strip(), begin_f, end_f))

            for value in node.values():
                _visit(value)
            return

        if isinstance(node, list):
            for item in node:
                _visit(item)

    _visit(payload)
    words.sort(key=lambda item: item[1])
    return words


def _map_words(
    *,
    canonical_words: list[CanonicalWord],
    predicted_words: list[tuple[str, float, float]],
    audio_duration_s: float,
) -> list[WordTiming]:
    matched_idx: dict[int, int] = {}
    matched_scores: dict[int, float] = {}
    cursor = 0

    norm_predicted = [
        (normalize_arabic(text), start_s, end_s)
        for text, start_s, end_s in predicted_words
    ]

    for index, canon in enumerate(canonical_words):
        best_j: int | None = None
        best_score = -1.0

        search_end = min(len(norm_predicted), cursor + 24)
        for predicted_index in range(cursor, search_end):
            text_norm, _, _ = norm_predicted[predicted_index]
            score = fuzz.ratio(canon.text_norm, text_norm)
            if score > best_score:
                best_score = score
                best_j = predicted_index
            if score >= 99:
                break

        if best_j is not None and best_score >= 55:
            matched_idx[index] = best_j
            matched_scores[index] = float(best_score)
            cursor = best_j + 1

    mapped: list[WordTiming] = []
    total = len(canonical_words)
    for index, canon in enumerate(canonical_words):
        pred_index = matched_idx.get(index)
        if pred_index is not None:
            _, start_s, end_s = norm_predicted[pred_index]
            alignment_origin = "native"
            match_score = matched_scores.get(index)
        else:
            start_s, end_s = _interpolate_slot(
                index=index,
                total=total,
                matched_idx=matched_idx,
                predicted_words=norm_predicted,
                audio_duration_s=audio_duration_s,
            )
            alignment_origin = "interpolated"
            match_score = None

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
                confidence=None,
                alignment_origin=alignment_origin,
                match_score=match_score,
                engine_candidate="mfa",
            )
        )

    return mapped


def _interpolate_slot(
    *,
    index: int,
    total: int,
    matched_idx: dict[int, int],
    predicted_words: list[tuple[str, float, float]],
    audio_duration_s: float,
) -> tuple[float, float]:
    prev_i = max((i for i in matched_idx if i < index), default=None)
    next_i = min((i for i in matched_idx if i > index), default=None)

    if prev_i is not None:
        left = predicted_words[matched_idx[prev_i]][2]
    else:
        left = 0.0

    if next_i is not None:
        right = predicted_words[matched_idx[next_i]][1]
    else:
        right = max(audio_duration_s, predicted_words[-1][2])

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
