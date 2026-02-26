from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
from typing import Any, Literal

import orjson
import soundfile as sf

from quran_audio_data.alignment import (
    AlignmentError,
    AlignmentOutput,
    EngineUnavailable,
    NemoAligner,
    WhisperXFallbackAligner,
)
from quran_audio_data.schema import (
    AudioMetadata,
    AyahTiming,
    EngineInfo,
    MetaInfo,
    QCThresholds,
    TimingResult,
    WordTiming,
    compute_qc,
    qc_requires_fallback,
)
from quran_audio_data.sources import ExistingTimingResolver
from quran_audio_data.text import CanonicalWord, QuranTextStore


DeviceOption = Literal["auto", "cpu", "cuda"]
EngineOption = Literal["nemo", "whisperx"]


class PipelineError(RuntimeError):
    pass


@dataclass(slots=True)
class ManifestRow:
    audio_path: Path
    reciter_id: str
    surah: int
    ayah: int | None
    source_url: str | None
    sha256: str | None
    language: str


@dataclass(slots=True)
class ProcessedFile:
    row: ManifestRow
    output_json: Path
    output_ayah_csv: Path
    output_words_csv: Path
    qc_report_json: Path
    source: str
    fallback_used: bool
    elapsed_s: float


@dataclass(slots=True)
class ProcessingSummary:
    total: int
    succeeded: int
    failed: int
    existing_resolved: int
    aligned: int
    fallback_used: int
    elapsed_s: float
    outputs: list[ProcessedFile]
    errors: list[str]


def read_manifest(manifest_path: str | Path) -> list[ManifestRow]:
    path = Path(manifest_path)
    if not path.exists():
        raise PipelineError(f"Manifest file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    required = {"audio_path", "reciter_id", "surah", "ayah"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise PipelineError(
            "Manifest missing required columns: " + ", ".join(sorted(missing))
        )

    parsed: list[ManifestRow] = []
    for idx, row in enumerate(rows, start=2):
        audio_path = Path(str(row.get("audio_path", "")).strip())
        reciter_id = str(row.get("reciter_id", "")).strip()
        if not reciter_id:
            raise PipelineError(f"manifest row {idx}: reciter_id is required")

        surah = _parse_int(row.get("surah"), f"manifest row {idx}: invalid surah")

        raw_ayah = str(row.get("ayah", "")).strip()
        ayah = None if raw_ayah == "" else _parse_int(raw_ayah, f"manifest row {idx}: invalid ayah")

        parsed.append(
            ManifestRow(
                audio_path=audio_path,
                reciter_id=reciter_id,
                surah=surah,
                ayah=ayah,
                source_url=_none_if_blank(row.get("source_url")),
                sha256=_none_if_blank(row.get("sha256")),
                language=_none_if_blank(row.get("language")) or "ar",
            )
        )

    return parsed


def run_alignment_pipeline(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    engine: EngineOption = "nemo",
    device: DeviceOption = "auto",
    text_data: str | Path | None = None,
    cache_dir: str | Path = ".cache/timings",
    enable_remote: bool = True,
    sample_size: int | None = None,
    thresholds: QCThresholds | None = None,
) -> ProcessingSummary:
    started = time.time()
    thresholds = thresholds or QCThresholds()
    rows = read_manifest(manifest_path)
    if sample_size is not None:
        rows = rows[:sample_size]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    text_store = QuranTextStore(text_data)
    resolver = ExistingTimingResolver(cache_dir=cache_dir, enable_remote=enable_remote)
    nemo = NemoAligner()
    whisperx = WhisperXFallbackAligner()

    outputs: list[ProcessedFile] = []
    errors: list[str] = []
    existing_resolved = 0
    aligned = 0
    fallback_used_count = 0

    for row in rows:
        try:
            processed = _process_row(
                row=row,
                out_dir=out_root,
                text_store=text_store,
                resolver=resolver,
                nemo=nemo,
                whisperx=whisperx,
                requested_engine=engine,
                device=device,
                thresholds=thresholds,
                cache_dir=cache_dir,
            )
            outputs.append(processed)
            if processed.source == "existing":
                existing_resolved += 1
            else:
                aligned += 1
            if processed.fallback_used:
                fallback_used_count += 1
        except Exception as exc:
            key = f"{row.reciter_id}:{row.surah}:{row.ayah or 'full'}"
            errors.append(f"{key}: {exc}")

    elapsed = time.time() - started
    return ProcessingSummary(
        total=len(rows),
        succeeded=len(outputs),
        failed=len(errors),
        existing_resolved=existing_resolved,
        aligned=aligned,
        fallback_used=fallback_used_count,
        elapsed_s=elapsed,
        outputs=outputs,
        errors=errors,
    )


def run_resolve_existing_only(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    text_data: str | Path | None = None,
    cache_dir: str | Path = ".cache/timings",
    enable_remote: bool = True,
    sample_size: int | None = None,
) -> ProcessingSummary:
    started = time.time()
    rows = read_manifest(manifest_path)
    if sample_size is not None:
        rows = rows[:sample_size]

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    text_store = QuranTextStore(text_data)
    resolver = ExistingTimingResolver(cache_dir=cache_dir, enable_remote=enable_remote)

    outputs: list[ProcessedFile] = []
    errors: list[str] = []

    for row in rows:
        try:
            audio_info = probe_audio(row.audio_path)
            canonical_words = _load_canonical_words(text_store=text_store, row=row)
            resolved = resolver.resolve(
                reciter_id=row.reciter_id,
                surah=row.surah,
                ayah=row.ayah,
                canonical_words=canonical_words,
                audio_duration_s=audio_info.duration_s,
                source_url=row.source_url,
            )
            if resolved is None:
                raise PipelineError("existing timing not found or failed validation")

            result = _build_result(
                row=row,
                audio_info=audio_info,
                engine_name="existing",
                engine_model=resolved.source_name,
                device="n/a",
                fallback_used=False,
                ayahs=resolved.ayahs,
                words=resolved.words,
                expected_word_count=len(canonical_words),
                thresholds=QCThresholds(),
            )
            outputs.append(_write_result_artifacts(result=result, row=row, out_dir=out_root, source="existing"))
        except Exception as exc:
            key = f"{row.reciter_id}:{row.surah}:{row.ayah or 'full'}"
            errors.append(f"{key}: {exc}")

    elapsed = time.time() - started
    return ProcessingSummary(
        total=len(rows),
        succeeded=len(outputs),
        failed=len(errors),
        existing_resolved=len(outputs),
        aligned=0,
        fallback_used=0,
        elapsed_s=elapsed,
        outputs=outputs,
        errors=errors,
    )


def validate_outputs(input_path: str | Path) -> tuple[int, int, list[str]]:
    root = Path(input_path)
    files: list[Path]
    if root.is_dir():
        files = sorted(root.glob("*.json"))
    else:
        files = [root]

    valid = 0
    invalid = 0
    errors: list[str] = []

    for file_path in files:
        # Skip qc companion report files.
        if file_path.name.endswith("_qc_report.json"):
            continue
        try:
            TimingResult.read_json(file_path)
            valid += 1
        except Exception as exc:
            invalid += 1
            errors.append(f"{file_path}: {exc}")

    return valid, invalid, errors


def benchmark_pipeline(
    *,
    manifest_path: str | Path,
    out_dir: str | Path,
    sample_size: int,
    engine: EngineOption = "nemo",
    device: DeviceOption = "auto",
    text_data: str | Path | None = None,
    cache_dir: str | Path = ".cache/timings",
    enable_remote: bool = True,
) -> dict[str, Any]:
    start = time.time()
    summary = run_alignment_pipeline(
        manifest_path=manifest_path,
        out_dir=out_dir,
        engine=engine,
        device=device,
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=enable_remote,
        sample_size=sample_size,
    )
    elapsed = time.time() - start

    avg_runtime = (
        sum(item.elapsed_s for item in summary.outputs) / len(summary.outputs)
        if summary.outputs
        else 0.0
    )

    return {
        "sample_size": sample_size,
        "total": summary.total,
        "succeeded": summary.succeeded,
        "failed": summary.failed,
        "existing_resolved": summary.existing_resolved,
        "aligned": summary.aligned,
        "fallback_used": summary.fallback_used,
        "elapsed_s": elapsed,
        "avg_file_runtime_s": avg_runtime,
        "errors": summary.errors,
    }


def _process_row(
    *,
    row: ManifestRow,
    out_dir: Path,
    text_store: QuranTextStore,
    resolver: ExistingTimingResolver,
    nemo: NemoAligner,
    whisperx: WhisperXFallbackAligner,
    requested_engine: EngineOption,
    device: DeviceOption,
    thresholds: QCThresholds,
    cache_dir: str | Path,
) -> ProcessedFile:
    started = time.time()

    audio_info = probe_audio(row.audio_path)
    canonical_words = _load_canonical_words(text_store=text_store, row=row)

    resolved = resolver.resolve(
        reciter_id=row.reciter_id,
        surah=row.surah,
        ayah=row.ayah,
        canonical_words=canonical_words,
        audio_duration_s=audio_info.duration_s,
        source_url=row.source_url,
    )

    if resolved is not None:
        result = _build_result(
            row=row,
            audio_info=audio_info,
            engine_name="existing",
            engine_model=resolved.source_name,
            device="n/a",
            fallback_used=False,
            ayahs=resolved.ayahs,
            words=resolved.words,
            expected_word_count=len(canonical_words),
            thresholds=thresholds,
        )
        return _write_result_artifacts(
            result=result,
            row=row,
            out_dir=out_dir,
            source="existing",
            fallback_used=False,
            elapsed_s=time.time() - started,
        )

    wav_path = ensure_wav_16k_mono(row.audio_path)
    fallback_used = False
    refinement_warnings: list[str] = []

    output = None
    if requested_engine == "nemo":
        try:
            if row.ayah is None:
                output, refinement_warnings = _align_full_surah_quality(
                    wav_path=wav_path,
                    canonical_words=canonical_words,
                    nemo=nemo,
                    audio_duration_s=audio_info.duration_s,
                    device=device,
                )
            else:
                output = nemo.align(
                    audio_wav_path=str(wav_path),
                    canonical_words=canonical_words,
                    audio_duration_s=audio_info.duration_s,
                    device=device,
                )
        except (EngineUnavailable, AlignmentError):
            output = whisperx.align(
                audio_wav_path=str(wav_path),
                canonical_words=canonical_words,
                audio_duration_s=audio_info.duration_s,
                device=device,
            )
            fallback_used = True
    else:
        output = whisperx.align(
            audio_wav_path=str(wav_path),
            canonical_words=canonical_words,
            audio_duration_s=audio_info.duration_s,
            device=device,
        )

    result = _build_result(
        row=row,
        audio_info=audio_info,
        engine_name=output.engine_name,
        engine_model=output.engine_model,
        device=output.device,
        fallback_used=fallback_used,
        ayahs=output.ayahs,
        words=output.words,
        expected_word_count=len(canonical_words),
        thresholds=thresholds,
    )
    result.qc.warnings.extend(refinement_warnings)

    if qc_requires_fallback(result.qc, thresholds) and output.engine_name != "whisperx":
        try:
            fallback_output = whisperx.align(
                audio_wav_path=str(wav_path),
                canonical_words=canonical_words,
                audio_duration_s=audio_info.duration_s,
                device=device,
            )
            fallback_used = True
            result = _build_result(
                row=row,
                audio_info=audio_info,
                engine_name=fallback_output.engine_name,
                engine_model=fallback_output.engine_model,
                device=fallback_output.device,
                fallback_used=fallback_used,
                ayahs=fallback_output.ayahs,
                words=fallback_output.words,
                expected_word_count=len(canonical_words),
                thresholds=thresholds,
            )
        except (AlignmentError, EngineUnavailable) as exc:
            result.qc.warnings.append(f"fallback_unavailable: {exc}")

    _write_cache_result(row=row, result=result, cache_root=cache_dir)

    return _write_result_artifacts(
        result=result,
        row=row,
        out_dir=out_dir,
        source=result.ayahs[0].source if result.ayahs else "aligned",
        fallback_used=fallback_used,
        elapsed_s=time.time() - started,
    )


def _align_full_surah_quality(
    *,
    wav_path: Path,
    canonical_words: list[CanonicalWord],
    nemo: NemoAligner,
    audio_duration_s: float,
    device: DeviceOption,
    max_passes: int = 3,
    base_overlap_s: float = 0.9,
) -> tuple[AlignmentOutput, list[str]]:
    """High-accuracy mode for full-surah files.

    Strategy:
    1. Full-surah forced alignment pass.
    2. Detect weak ayahs (collapsed/invalid timings).
    3. Re-align weak ayahs on overlapped chunks, iteratively.
    """

    warnings: list[str] = []

    primary = nemo.align(
        audio_wav_path=str(wav_path),
        canonical_words=canonical_words,
        audio_duration_s=audio_duration_s,
        device=device,
    )
    refined_words = sorted(primary.words, key=lambda word: word.word_index_global)

    expected_by_ayah: dict[int, list[CanonicalWord]] = defaultdict(list)
    for canon in canonical_words:
        expected_by_ayah[canon.ayah].append(canon)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        warnings.append("quality_refine_skipped: ffmpeg_unavailable")
        ayahs = _derive_ayahs_from_words(refined_words, source="aligned")
        return (
            AlignmentOutput(
                ayahs=ayahs,
                words=refined_words,
                engine_name="nemo",
                engine_model=nemo.model_name,
                device=primary.device,
                source="aligned",
            ),
            warnings,
        )

    with tempfile.TemporaryDirectory(prefix="qad_refine_") as temp_dir:
        temp_root = Path(temp_dir)
        for pass_index in range(max_passes):
            weak_ayahs = _find_weak_ayahs(
                words=refined_words,
                expected_by_ayah=expected_by_ayah,
            )
            if not weak_ayahs:
                break

            overlap_s = base_overlap_s * (pass_index + 1)
            warnings.append(
                f"quality_refine_pass_{pass_index + 1}: weak_ayahs={','.join(str(v) for v in weak_ayahs)} overlap_s={overlap_s:.2f}"
            )

            any_updated = False
            current_by_ayah = _words_by_ayah(refined_words)

            for ayah in weak_ayahs:
                expected_words = expected_by_ayah.get(ayah, [])
                if not expected_words:
                    continue

                current_words = current_by_ayah.get(ayah, [])
                if current_words:
                    base_start = min(word.start_s for word in current_words)
                    base_end = max(word.end_s for word in current_words)
                else:
                    ratio_start, ratio_end = _estimate_ayah_ratio_window(
                        ayah=ayah,
                        expected_by_ayah=expected_by_ayah,
                    )
                    base_start = ratio_start * audio_duration_s
                    base_end = ratio_end * audio_duration_s

                # Ensure we always provide a usable chunk.
                if base_end <= base_start:
                    base_end = min(audio_duration_s, base_start + 2.0)

                chunk_start = max(0.0, base_start - overlap_s)
                chunk_end = min(audio_duration_s, base_end + overlap_s)
                if chunk_end <= chunk_start:
                    continue

                chunk_path = temp_root / f"ayah_{ayah:03d}_pass_{pass_index + 1}.wav"
                _cut_audio_chunk(
                    ffmpeg=ffmpeg,
                    src_wav=wav_path,
                    dst_wav=chunk_path,
                    start_s=chunk_start,
                    end_s=chunk_end,
                )

                chunk_output = nemo.align(
                    audio_wav_path=str(chunk_path),
                    canonical_words=expected_words,
                    audio_duration_s=(chunk_end - chunk_start),
                    device=device,
                )

                shifted = [
                    word.model_copy(
                        update={
                            "start_s": word.start_s + chunk_start,
                            "end_s": word.end_s + chunk_start,
                        }
                    )
                    for word in chunk_output.words
                ]

                refined_words = [word for word in refined_words if word.ayah != ayah]
                refined_words.extend(shifted)
                refined_words.sort(key=lambda word: word.word_index_global)
                any_updated = True

            if not any_updated:
                break

    ayahs = _derive_ayahs_from_words(refined_words, source="aligned")
    return (
        AlignmentOutput(
            ayahs=ayahs,
            words=refined_words,
            engine_name="nemo",
            engine_model=nemo.model_name,
            device=primary.device,
            source="aligned",
        ),
        warnings,
    )


def _find_weak_ayahs(
    *,
    words: list[WordTiming],
    expected_by_ayah: dict[int, list[CanonicalWord]],
) -> list[int]:
    by_ayah = _words_by_ayah(words)
    weak: list[int] = []

    for ayah in sorted(expected_by_ayah):
        expected_count = len(expected_by_ayah[ayah])
        actual_words = by_ayah.get(ayah, [])

        if len(actual_words) != expected_count:
            weak.append(ayah)
            continue

        if not actual_words:
            weak.append(ayah)
            continue

        starts = [word.start_s for word in actual_words]
        if starts != sorted(starts):
            weak.append(ayah)
            continue

        non_positive = sum(1 for word in actual_words if word.end_s <= word.start_s)
        if non_positive > 0:
            weak.append(ayah)
            continue

        tiny = sum(1 for word in actual_words if (word.end_s - word.start_s) <= 0.03)
        if tiny / len(actual_words) > 0.12:
            weak.append(ayah)

    return weak


def _words_by_ayah(words: list[WordTiming]) -> dict[int, list[WordTiming]]:
    out: dict[int, list[WordTiming]] = defaultdict(list)
    for word in words:
        out[word.ayah].append(word)
    for ayah in out:
        out[ayah].sort(key=lambda word: word.word_index_in_ayah)
    return out


def _estimate_ayah_ratio_window(
    *,
    ayah: int,
    expected_by_ayah: dict[int, list[CanonicalWord]],
) -> tuple[float, float]:
    total = sum(len(group) for group in expected_by_ayah.values())
    if total <= 0:
        return 0.0, 1.0

    before = sum(len(expected_by_ayah[key]) for key in sorted(expected_by_ayah) if key < ayah)
    current = len(expected_by_ayah.get(ayah, []))
    start_ratio = before / total
    end_ratio = (before + current) / total
    return start_ratio, end_ratio


def _cut_audio_chunk(
    *,
    ffmpeg: str,
    src_wav: Path,
    dst_wav: Path,
    start_s: float,
    end_s: float,
) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(src_wav),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PipelineError(f"ffmpeg chunk extraction failed: {proc.stderr.strip()}")


def _derive_ayahs_from_words(words: list[WordTiming], *, source: str) -> list[AyahTiming]:
    grouped: dict[tuple[int, int], list[WordTiming]] = defaultdict(list)
    for word in words:
        grouped[(word.surah, word.ayah)].append(word)

    ayahs: list[AyahTiming] = []
    for (surah, ayah), group in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        ayahs.append(
            AyahTiming(
                surah=surah,
                ayah=ayah,
                start_s=min(item.start_s for item in group),
                end_s=max(item.end_s for item in group),
                source=source,
            )
        )
    return ayahs


def _load_canonical_words(*, text_store: QuranTextStore, row: ManifestRow) -> list[CanonicalWord]:
    words = text_store.build_words(surah=row.surah, ayah=row.ayah)
    if not words:
        raise PipelineError(f"No canonical words found for surah={row.surah} ayah={row.ayah}")
    return words


def _build_result(
    *,
    row: ManifestRow,
    audio_info: AudioMetadata,
    engine_name: str,
    engine_model: str,
    device: str,
    fallback_used: bool,
    ayahs,
    words,
    expected_word_count: int,
    thresholds: QCThresholds,
) -> TimingResult:
    input_mode = "ayah_file" if row.ayah is not None else "full_surah"
    qc = compute_qc(
        words=words,
        expected_word_count=expected_word_count,
        audio_duration_s=audio_info.duration_s,
        thresholds=thresholds,
    )

    return TimingResult(
        audio=audio_info,
        meta=MetaInfo(reciter_id=row.reciter_id, surah=row.surah, input_mode=input_mode),
        engine=EngineInfo(
            name=engine_name,
            model=engine_model,
            device=device,
            fallback_used=fallback_used,
        ),
        ayahs=ayahs,
        words=words,
        qc=qc,
    )


def _write_result_artifacts(
    *,
    result: TimingResult,
    row: ManifestRow,
    out_dir: Path,
    source: str,
    fallback_used: bool = False,
    elapsed_s: float = 0.0,
) -> ProcessedFile:
    stem = _output_stem(row)
    json_path = out_dir / f"{stem}.json"
    result.write_json(json_path)
    ayah_csv, words_csv = result.write_csvs(out_dir / stem)

    qc_path = out_dir / f"{stem}_qc_report.json"
    qc_path.write_bytes(orjson.dumps(result.qc.model_dump(mode="json"), option=orjson.OPT_INDENT_2))

    return ProcessedFile(
        row=row,
        output_json=json_path,
        output_ayah_csv=ayah_csv,
        output_words_csv=words_csv,
        qc_report_json=qc_path,
        source=source,
        fallback_used=fallback_used,
        elapsed_s=elapsed_s,
    )


def _write_cache_result(
    *,
    row: ManifestRow,
    result: TimingResult,
    cache_root: str | Path = ".cache/timings",
) -> None:
    cache_dir = Path(cache_root) / row.reciter_id
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / f"{row.surah:03d}.json"
    path.write_bytes(result.to_json_bytes())

    if row.ayah is not None:
        ayah_path = cache_dir / f"{row.surah:03d}_{row.ayah:03d}.json"
        ayah_path.write_bytes(result.to_json_bytes())


def _output_stem(row: ManifestRow) -> str:
    if row.ayah is None:
        return f"{row.reciter_id}_s{row.surah:03d}_full"
    return f"{row.reciter_id}_s{row.surah:03d}_a{row.ayah:03d}"


def _none_if_blank(value: Any) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _parse_int(value: Any, message: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PipelineError(message) from exc


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_audio(path: str | Path) -> AudioMetadata:
    file_path = Path(path)
    if not file_path.exists():
        raise PipelineError(f"Audio file not found: {file_path}")

    try:
        info = sf.info(str(file_path))
        return AudioMetadata(
            path=str(file_path),
            duration_s=float(info.duration),
            sample_rate=int(info.samplerate),
            channels=int(info.channels),
        )
    except RuntimeError:
        pass

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        raise PipelineError(
            f"Unable to probe {file_path}. Install ffprobe or use wav/flac readable by soundfile."
        )

    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        str(file_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PipelineError(f"ffprobe failed for {file_path}: {proc.stderr.strip()}")

    payload = json.loads(proc.stdout)
    streams = payload.get("streams", [])
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    format_obj = payload.get("format", {})

    duration_s = float(format_obj.get("duration") or audio_stream.get("duration") or 0.0)
    sample_rate = int(audio_stream.get("sample_rate") or 16000)
    channels = int(audio_stream.get("channels") or 1)

    return AudioMetadata(
        path=str(file_path),
        duration_s=duration_s,
        sample_rate=sample_rate,
        channels=channels,
    )


def ensure_wav_16k_mono(path: str | Path) -> Path:
    src = Path(path)

    try:
        info = sf.info(str(src))
        if src.suffix.lower() == ".wav" and info.samplerate == 16000 and info.channels == 1:
            return src
    except RuntimeError:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PipelineError(
            "ffmpeg is required for converting input audio to mono WAV 16k."
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix="qad_audio_"))
    dst = tmp_dir / f"{src.stem}_16k_mono.wav"

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        "16000",
        str(dst),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PipelineError(f"ffmpeg conversion failed: {proc.stderr.strip()}")

    return dst
