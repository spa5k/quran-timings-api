from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import orjson
import soundfile as sf
import pytest

import quran_audio_data.pipeline as pipeline
from quran_audio_data.alignment.base import AlignmentOutput
from quran_audio_data.schema import (
    AudioMetadata,
    AyahTiming,
    EngineInfo,
    MetaInfo,
    QCReport,
    TimingResult,
    WordTiming,
)
from quran_audio_data.text import QuranTextStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_DATA = PROJECT_ROOT / "data" / "quran_text_uthmani_v1.json"


class _AlwaysAvailableMFA:
    def is_available(self) -> bool:
        return True

    def availability_error(self) -> str:
        return ""

    def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
        words = []
        total = max(1, len(canonical_words))
        for idx, word in enumerate(canonical_words):
            start = idx / total
            end = (idx + 1) / total
            words.append(
                WordTiming(
                    surah=word.surah,
                    ayah=word.ayah,
                    word_index_global=word.word_index_global,
                    word_index_in_ayah=word.word_index_in_ayah,
                    text_uthmani=word.text_uthmani,
                    text_norm=word.text_norm,
                    start_s=start,
                    end_s=end,
                    confidence=0.4,
                    alignment_origin="interpolated",
                    match_score=40.0,
                    engine_candidate="mfa",
                )
            )
        ayahs = [AyahTiming(surah=canonical_words[0].surah, ayah=canonical_words[0].ayah, start_s=0.0, end_s=1.0, source="aligned")]
        return AlignmentOutput(
            ayahs=ayahs,
            words=words,
            engine_name="mfa",
            engine_model="fake-mfa",
            device="cpu",
            source="aligned",
        )


class _AlwaysAvailableNemo:
    model_name = "fake-nemo"

    def __init__(self, *args, **kwargs) -> None:
        pass

    def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
        words = []
        total = max(1, len(canonical_words))
        for idx, word in enumerate(canonical_words):
            start = idx / total
            end = (idx + 1) / total
            words.append(
                WordTiming(
                    surah=word.surah,
                    ayah=word.ayah,
                    word_index_global=word.word_index_global,
                    word_index_in_ayah=word.word_index_in_ayah,
                    text_uthmani=word.text_uthmani,
                    text_norm=word.text_norm,
                    start_s=start,
                    end_s=end,
                    confidence=0.95,
                    alignment_origin="native",
                    match_score=99.0,
                    engine_candidate="nemo",
                )
            )
        ayahs = [
            AyahTiming(
                surah=canonical_words[0].surah,
                ayah=canonical_words[0].ayah,
                start_s=0.0,
                end_s=1.0,
                source="aligned",
            )
        ]
        return AlignmentOutput(
            ayahs=ayahs,
            words=words,
            engine_name="nemo",
            engine_model="fake-nemo",
            device="cpu",
            source="aligned",
        )


class _UnavailableWhisperX:
    def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
        raise pipeline.EngineUnavailable("unused")


def _write_silence_wav(path: Path, duration_s: float = 1.0, sample_rate: int = 16000) -> None:
    samples = int(duration_s * sample_rate)
    data = np.zeros(samples, dtype=np.float32)
    sf.write(path, data, sample_rate)


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["audio_path", "reciter_id", "surah", "ayah", "source_url", "sha256", "language"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _build_existing_payload(store: QuranTextStore, *, surah: int, ayah: int | None) -> dict:
    canonical = store.build_words(surah=surah, ayah=ayah)
    total = len(canonical)

    words: list[dict] = []
    ayah_spans: dict[int, tuple[float, float]] = {}

    for idx, word in enumerate(canonical):
        start = idx / total
        end = (idx + 1) / total
        words.append(
            {
                "surah": word.surah,
                "ayah": word.ayah,
                "word_index_global": word.word_index_global,
                "word_index_in_ayah": word.word_index_in_ayah,
                "text_uthmani": word.text_uthmani,
                "text_norm": word.text_norm,
                "start_s": start,
                "end_s": end,
                "confidence": 0.99,
                "alignment_origin": "native",
                "match_score": 100.0,
            }
        )

        if word.ayah not in ayah_spans:
            ayah_spans[word.ayah] = (start, end)
        else:
            old = ayah_spans[word.ayah]
            ayah_spans[word.ayah] = (old[0], end)

    ayahs = [
        {
            "surah": surah,
            "ayah": ayah_num,
            "start_s": span[0],
            "end_s": span[1],
            "source": "existing",
        }
        for ayah_num, span in sorted(ayah_spans.items())
    ]

    return {
        "schema_version": "v1",
        "audio": {"path": "n/a", "duration_s": 1.0, "sample_rate": 16000, "channels": 1},
        "meta": {"reciter_id": "test", "surah": surah, "input_mode": "ayah_file" if ayah else "full_surah"},
        "engine": {"name": "existing", "model": "cache", "device": "n/a", "fallback_used": False},
        "ayahs": ayahs,
        "words": words,
        "qc": {"coverage": 1.0, "monotonic": True, "duration_match": True, "warnings": []},
    }


def test_ayah_file_mode_resolves_existing_and_exports(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "NemoAligner", _AlwaysAvailableNemo)
    monkeypatch.setattr(pipeline, "WhisperXFallbackAligner", _UnavailableWhisperX)
    monkeypatch.setattr(pipeline, "MFAAligner", _AlwaysAvailableMFA)

    store = QuranTextStore(TEXT_DATA)
    audio = tmp_path / "ayah.wav"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / ".cache" / "timings" / "test_reciter"

    _write_silence_wav(audio)
    _write_manifest(
        manifest,
        [
            {
                "audio_path": str(audio),
                "reciter_id": "test_reciter",
                "surah": "1",
                "ayah": "1",
                "source_url": "",
                "sha256": "",
                "language": "ar",
            }
        ],
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = _build_existing_payload(store, surah=1, ayah=1)
    (cache_dir / "001_001.json").write_bytes(orjson.dumps(payload))

    summary = pipeline.run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out_dir,
        text_data=TEXT_DATA,
        cache_dir=tmp_path / ".cache" / "timings",
        enable_remote=False,
    )

    assert summary.succeeded == 1
    assert summary.priors_used == 1
    assert summary.aligned == 1
    assert summary.failed == 0

    output_json = out_dir / "test_reciter_s001_a001.json"
    assert output_json.exists()


def test_full_surah_mode_resolves_existing_for_all_ayahs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "NemoAligner", _AlwaysAvailableNemo)
    monkeypatch.setattr(pipeline, "WhisperXFallbackAligner", _UnavailableWhisperX)
    monkeypatch.setattr(pipeline, "MFAAligner", _AlwaysAvailableMFA)

    store = QuranTextStore(TEXT_DATA)
    audio = tmp_path / "surah.wav"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"
    cache_dir = tmp_path / ".cache" / "timings" / "full_reciter"

    _write_silence_wav(audio)
    _write_manifest(
        manifest,
        [
            {
                "audio_path": str(audio),
                "reciter_id": "full_reciter",
                "surah": "1",
                "ayah": "",
                "source_url": "",
                "sha256": "",
                "language": "ar",
            }
        ],
    )

    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = _build_existing_payload(store, surah=1, ayah=None)
    (cache_dir / "001.json").write_bytes(orjson.dumps(payload))

    summary = pipeline.run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out_dir,
        text_data=TEXT_DATA,
        cache_dir=tmp_path / ".cache" / "timings",
        enable_remote=False,
    )

    assert summary.succeeded == 1
    assert summary.priors_used == 1
    result = orjson.loads((out_dir / "full_reciter_s001_full.json").read_bytes())
    assert len(result["ayahs"]) == 7


def test_missing_existing_triggers_fallback_and_outputs_valid_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pipeline, "MFAAligner", _AlwaysAvailableMFA)

    audio = tmp_path / "surah.wav"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"

    _write_silence_wav(audio)
    _write_manifest(
        manifest,
        [
            {
                "audio_path": str(audio),
                "reciter_id": "fallback_reciter",
                "surah": "1",
                "ayah": "1",
                "source_url": "",
                "sha256": "",
                "language": "ar",
            }
        ],
    )

    class FakeNemo:
        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            words = []
            for idx, word in enumerate(canonical_words):
                # Non-monotonic starts force QC fallback.
                start = 0.6 - (idx * 0.1)
                end = start + 0.05
                words.append(
                    WordTiming(
                        surah=word.surah,
                        ayah=word.ayah,
                        word_index_global=word.word_index_global,
                        word_index_in_ayah=word.word_index_in_ayah,
                        text_uthmani=word.text_uthmani,
                        text_norm=word.text_norm,
                        start_s=start,
                        end_s=end,
                        confidence=0.9,
                    )
                )

            ayahs = [AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="aligned")]
            return AlignmentOutput(
                ayahs=ayahs,
                words=words,
                engine_name="nemo",
                engine_model="fake",
                device="cpu",
                source="aligned",
            )

    class FakeWhisperX:
        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            words = []
            total = len(canonical_words)
            for idx, word in enumerate(canonical_words):
                start = idx / total
                end = (idx + 1) / total
                words.append(
                    WordTiming(
                        surah=word.surah,
                        ayah=word.ayah,
                        word_index_global=word.word_index_global,
                        word_index_in_ayah=word.word_index_in_ayah,
                        text_uthmani=word.text_uthmani,
                        text_norm=word.text_norm,
                        start_s=start,
                        end_s=end,
                        confidence=0.95,
                    )
                )

            ayahs = [AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="fallback")]
            return AlignmentOutput(
                ayahs=ayahs,
                words=words,
                engine_name="whisperx",
                engine_model="fake",
                device="cpu",
                source="fallback",
            )

    monkeypatch.setattr(pipeline, "NemoAligner", FakeNemo)
    monkeypatch.setattr(pipeline, "WhisperXFallbackAligner", FakeWhisperX)

    summary = pipeline.run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out_dir,
        text_data=TEXT_DATA,
        cache_dir=tmp_path / ".cache" / "timings",
        enable_remote=False,
        engine="nemo",
        device="cpu",
    )

    assert summary.succeeded == 1
    assert summary.fallback_used == 1
    valid, invalid, errors = pipeline.validate_outputs(out_dir)
    assert valid == 1
    assert invalid == 0
    assert errors == []


def test_strict_mode_multi_engine_prefers_higher_quality_candidate(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    audio = tmp_path / "strict.wav"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"

    _write_silence_wav(audio)
    _write_manifest(
        manifest,
        [
            {
                "audio_path": str(audio),
                "reciter_id": "strict_reciter",
                "surah": "1",
                "ayah": "1",
                "source_url": "",
                "sha256": "",
                "language": "ar",
            }
        ],
    )

    class FakeNemo:
        model_name = "fake-nemo"

        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            total = len(canonical_words)
            words = []
            for idx, word in enumerate(canonical_words):
                start = idx / total
                end = (idx + 1) / total
                words.append(
                    WordTiming(
                        surah=word.surah,
                        ayah=word.ayah,
                        word_index_global=word.word_index_global,
                        word_index_in_ayah=word.word_index_in_ayah,
                        text_uthmani=word.text_uthmani,
                        text_norm=word.text_norm,
                        start_s=start,
                        end_s=end,
                        confidence=0.7,
                        alignment_origin="interpolated",
                        match_score=35.0,
                        engine_candidate="nemo",
                    )
                )
            return AlignmentOutput(
                ayahs=[AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="aligned")],
                words=words,
                engine_name="nemo",
                engine_model="fake-nemo",
                device="cpu",
                source="aligned",
            )

    class FakeWhisperX:
        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            total = len(canonical_words)
            words = []
            for idx, word in enumerate(canonical_words):
                start = idx / total
                end = (idx + 1) / total
                words.append(
                    WordTiming(
                        surah=word.surah,
                        ayah=word.ayah,
                        word_index_global=word.word_index_global,
                        word_index_in_ayah=word.word_index_in_ayah,
                        text_uthmani=word.text_uthmani,
                        text_norm=word.text_norm,
                        start_s=start,
                        end_s=end,
                        confidence=0.98,
                        alignment_origin="native",
                        match_score=99.0,
                        engine_candidate="whisperx",
                    )
                )
            return AlignmentOutput(
                ayahs=[AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="fallback")],
                words=words,
                engine_name="whisperx",
                engine_model="fake-whisperx",
                device="cpu",
                source="fallback",
            )

    class FakeMFA:
        def is_available(self) -> bool:
            return True

        def availability_error(self) -> str:
            return ""

        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            raise pipeline.EngineUnavailable("not expected")

    monkeypatch.setattr(pipeline, "NemoAligner", FakeNemo)
    monkeypatch.setattr(pipeline, "WhisperXFallbackAligner", FakeWhisperX)
    monkeypatch.setattr(pipeline, "MFAAligner", FakeMFA)

    summary = pipeline.run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out_dir,
        text_data=TEXT_DATA,
        cache_dir=tmp_path / ".cache" / "timings",
        enable_remote=False,
        engine="nemo",
        multi_engine=["nemo", "whisperx"],
        accuracy_mode="strict",
        device="cpu",
    )

    assert summary.succeeded == 1
    assert summary.fallback_used == 1
    payload = orjson.loads((out_dir / "strict_reciter_s001_a001.json").read_bytes())
    assert payload["engine"]["name"] == "ensemble"
    assert all(word["engine_candidate"] == "whisperx" for word in payload["words"])


def test_normalize_engines_always_includes_all_tracks() -> None:
    engines = pipeline._normalize_engines(
        requested_engine="nemo",
        multi_engine=["nemo"],
    )
    assert engines == ["nemo", "whisperx", "mfa"]


def test_non_strict_mode_is_rejected(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    audio = tmp_path / "all_bad.wav"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"

    _write_silence_wav(audio)
    _write_manifest(
        manifest,
        [
            {
                "audio_path": str(audio),
                "reciter_id": "all_bad_reciter",
                "surah": "1",
                "ayah": "1",
                "source_url": "",
                "sha256": "",
                "language": "ar",
            }
        ],
    )

    def _build_bad_words(canonical_words):
        words = []
        for idx, word in enumerate(canonical_words):
            start = 0.6 - (idx * 0.1)
            end = start + 0.05
            words.append(
                WordTiming(
                    surah=word.surah,
                    ayah=word.ayah,
                    word_index_global=word.word_index_global,
                    word_index_in_ayah=word.word_index_in_ayah,
                    text_uthmani=word.text_uthmani,
                    text_norm=word.text_norm,
                    start_s=start,
                    end_s=end,
                    confidence=0.4,
                    alignment_origin="native",
                    match_score=40.0,
                )
            )
        return words

    class FakeNemo:
        model_name = "fake-nemo"

        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            words = _build_bad_words(canonical_words)
            return AlignmentOutput(
                ayahs=[AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="aligned")],
                words=words,
                engine_name="nemo",
                engine_model="fake-nemo",
                device="cpu",
                source="aligned",
            )

    class FakeWhisperX:
        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            words = _build_bad_words(canonical_words)
            return AlignmentOutput(
                ayahs=[AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="fallback")],
                words=words,
                engine_name="whisperx",
                engine_model="fake-whisperx",
                device="cpu",
                source="fallback",
            )

    class FakeMFA:
        def is_available(self) -> bool:
            return True

        def availability_error(self) -> str:
            return ""

        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            words = _build_bad_words(canonical_words)
            return AlignmentOutput(
                ayahs=[AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="aligned")],
                words=words,
                engine_name="mfa",
                engine_model="fake-mfa",
                device="cpu",
                source="aligned",
            )

    monkeypatch.setattr(pipeline, "NemoAligner", FakeNemo)
    monkeypatch.setattr(pipeline, "WhisperXFallbackAligner", FakeWhisperX)
    monkeypatch.setattr(pipeline, "MFAAligner", FakeMFA)

    with pytest.raises(pipeline.PipelineError):
        pipeline.run_alignment_pipeline(
            manifest_path=manifest,
            out_dir=out_dir,
            text_data=TEXT_DATA,
            cache_dir=tmp_path / ".cache" / "timings",
            enable_remote=False,
            accuracy_mode="standard",  # type: ignore[arg-type]
            engine="nemo",
            device="cpu",
        )


def test_strict_mode_rejects_bad_refinement_and_keeps_original(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    audio = tmp_path / "strict_refine.wav"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"

    _write_silence_wav(audio)
    _write_manifest(
        manifest,
        [
            {
                "audio_path": str(audio),
                "reciter_id": "strict_refine_reciter",
                "surah": "1",
                "ayah": "1",
                "source_url": "",
                "sha256": "",
                "language": "ar",
            }
        ],
    )

    class FakeNemo:
        model_name = "fake-nemo"

        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            total = len(canonical_words)
            words = []
            for idx, word in enumerate(canonical_words):
                start = idx / total
                end = (idx + 1) / total
                words.append(
                    WordTiming(
                        surah=word.surah,
                        ayah=word.ayah,
                        word_index_global=word.word_index_global,
                        word_index_in_ayah=word.word_index_in_ayah,
                        text_uthmani=word.text_uthmani,
                        text_norm=word.text_norm,
                        start_s=start,
                        end_s=end,
                        confidence=0.99,
                        alignment_origin="native",
                        match_score=99.0,
                        engine_candidate="nemo",
                    )
                )
            return AlignmentOutput(
                ayahs=[AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="aligned")],
                words=words,
                engine_name="nemo",
                engine_model="fake-nemo",
                device="cpu",
                source="aligned",
            )

    class FakeWhisperX:
        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            raise pipeline.EngineUnavailable("unused")

    class FakeMFA:
        def is_available(self) -> bool:
            return True

        def availability_error(self) -> str:
            return ""

        def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
            raise pipeline.EngineUnavailable("unused")

    def fake_refine(*, words, wav_path, max_shift_s, min_duration_s):
        broken = []
        for word in words:
            broken.append(word.model_copy(update={"end_s": word.start_s}))
        return broken

    monkeypatch.setattr(pipeline, "NemoAligner", FakeNemo)
    monkeypatch.setattr(pipeline, "WhisperXFallbackAligner", FakeWhisperX)
    monkeypatch.setattr(pipeline, "MFAAligner", FakeMFA)
    monkeypatch.setattr(pipeline, "_refine_word_boundaries", fake_refine)

    summary = pipeline.run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out_dir,
        text_data=TEXT_DATA,
        cache_dir=tmp_path / ".cache" / "timings",
        enable_remote=False,
        accuracy_mode="strict",
        engine="nemo",
        device="cpu",
    )

    assert summary.succeeded == 1
    payload = orjson.loads((out_dir / "strict_refine_reciter_s001_a001.json").read_bytes())
    assert payload["qc"]["zero_or_negative_ratio"] == 0.0
    assert "boundary_refinement_rejected" in payload["qc"]["warnings"]


def test_ayah_cache_write_does_not_overwrite_surah_cache(tmp_path) -> None:
    row = pipeline.ManifestRow(
        audio_path=tmp_path / "sample.wav",
        reciter_id="cache_test",
        surah=1,
        ayah=1,
        source_url=None,
        sha256=None,
        language="ar",
        riwaya=None,
        text_variant=None,
        reference_split=None,
    )
    result = TimingResult(
        audio=AudioMetadata(path="sample.wav", duration_s=1.0, sample_rate=16000, channels=1),
        meta=MetaInfo(reciter_id="cache_test", surah=1, input_mode="ayah_file"),
        engine=EngineInfo(name="nemo", model="fake", device="cpu", fallback_used=False),
        ayahs=[AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="aligned")],
        words=[
            WordTiming(
                surah=1,
                ayah=1,
                word_index_global=1,
                word_index_in_ayah=1,
                text_uthmani="بِسْمِ",
                text_norm="بسم",
                start_s=0.0,
                end_s=0.5,
                confidence=0.95,
                alignment_origin="native",
                match_score=100.0,
                engine_candidate="nemo",
            )
        ],
        qc=QCReport(coverage=1.0, monotonic=True, duration_match=True, warnings=[]),
    )

    pipeline._write_cache_result(row=row, result=result, cache_root=tmp_path / ".cache")

    reciter_cache = tmp_path / ".cache" / "cache_test"
    assert (reciter_cache / "001_001.json").exists()
    assert not (reciter_cache / "001.json").exists()
