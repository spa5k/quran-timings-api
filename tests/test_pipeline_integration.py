from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import orjson
import soundfile as sf

import quran_audio_data.pipeline as pipeline
from quran_audio_data.alignment.base import AlignmentOutput
from quran_audio_data.schema import AyahTiming, WordTiming
from quran_audio_data.text import QuranTextStore


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_DATA = PROJECT_ROOT / "data" / "quran_text_uthmani_v1.json"


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
    assert summary.existing_resolved == 1
    assert summary.failed == 0

    output_json = out_dir / "test_reciter_s001_a001.json"
    assert output_json.exists()


def test_full_surah_mode_resolves_existing_for_all_ayahs(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

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
    assert summary.existing_resolved == 1
    result = orjson.loads((out_dir / "full_reciter_s001_full.json").read_bytes())
    assert len(result["ayahs"]) == 7


def test_missing_existing_triggers_fallback_and_outputs_valid_schema(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

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
