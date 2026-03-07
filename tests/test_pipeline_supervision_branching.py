from __future__ import annotations

from pathlib import Path
import csv

import numpy as np
import orjson
import soundfile as sf

import quran_audio_data.pipeline as pipeline
import quran_audio_data.pipeline.orchestrator as orchestrator
from quran_audio_data.alignment.base import AlignmentOutput
from quran_audio_data.schema import AyahTiming, WordTiming
from quran_audio_data.supervision.qcom_audio import VerseSegmentPayload
from quran_audio_data.supervision.segment_normalizer import WordSegment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_DATA = PROJECT_ROOT / "data" / "quran_text_uthmani_v1.json"


def _write_silence(path: Path) -> None:
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)


def _write_manifest(path: Path, reciter_id: str) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "audio_path",
                "reciter_id",
                "surah",
                "ayah",
                "source_url",
                "sha256",
                "language",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": str(path.parent / "sample.wav"),
                "reciter_id": reciter_id,
                "surah": "1",
                "ayah": "1",
                "source_url": "",
                "sha256": "",
                "language": "ar",
            }
        )


class FakeNemo:
    model_name = "fake"

    def __init__(self, *args, **kwargs) -> None:
        pass

    def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
        total = len(canonical_words)
        words = []
        for idx, word in enumerate(canonical_words):
            words.append(
                WordTiming(
                    surah=word.surah,
                    ayah=word.ayah,
                    word_index_global=word.word_index_global,
                    word_index_in_ayah=word.word_index_in_ayah,
                    text_uthmani=word.text_uthmani,
                    text_norm=word.text_norm,
                    start_s=idx / total,
                    end_s=(idx + 1) / total,
                    confidence=0.9,
                    alignment_origin="native",
                    match_score=95.0,
                    engine_candidate="nemo",
                )
            )
        return AlignmentOutput(
            ayahs=[AyahTiming(surah=1, ayah=1, start_s=0.0, end_s=1.0, source="aligned")],
            words=words,
            engine_name="nemo",
            engine_model="fake",
            device="cpu",
            source="aligned",
        )


class FakeWhisperX:
    def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):
        raise pipeline.EngineUnavailable("unused")


def test_unsupported_reciter_uses_model_only_supervision_path(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "sample.wav"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"
    _write_silence(audio)
    _write_manifest(manifest, "yasser_ad-dussary")

    monkeypatch.setattr(pipeline, "NemoAligner", FakeNemo)
    monkeypatch.setattr(pipeline, "WhisperXFallbackAligner", FakeWhisperX)

    summary = pipeline.run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out_dir,
        text_data=TEXT_DATA,
        cache_dir=tmp_path / ".cache" / "timings",
        enable_remote=False,
    )

    assert summary.succeeded == 1
    payload = orjson.loads((out_dir / "yasser_ad-dussary_s001_a001.json").read_bytes())
    assert payload["segment_source_type"] == "none"


def test_supported_reciter_injects_qcom_word_supervision(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "sample.wav"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"
    _write_silence(audio)
    _write_manifest(manifest, "abdurrahmaan_as-sudays")

    monkeypatch.setattr(pipeline, "NemoAligner", FakeNemo)
    monkeypatch.setattr(pipeline, "WhisperXFallbackAligner", FakeWhisperX)

    def fake_fetch_best_verse_segments(*, recitation_id, chapter, verse_key):
        return VerseSegmentPayload(
            verse_key=verse_key,
            segments=[
                WordSegment(word_index=1, start_ms=0.0, end_ms=120.0),
                WordSegment(word_index=2, start_ms=130.0, end_ms=260.0),
            ],
            source_type="qcom_verse",
            segment_shape="4_field",
        )

    monkeypatch.setattr(orchestrator, "fetch_best_verse_segments", fake_fetch_best_verse_segments)

    summary = pipeline.run_alignment_pipeline(
        manifest_path=manifest,
        out_dir=out_dir,
        text_data=TEXT_DATA,
        cache_dir=tmp_path / ".cache" / "timings",
        enable_remote=True,
    )

    assert summary.succeeded == 1
    payload = orjson.loads((out_dir / "abdurrahmaan_as-sudays_s001_a001.json").read_bytes())
    assert payload["segment_source_type"] == "qcom_verse"
    assert any("shape=4_field" in source for source in payload["supervision_sources"])
