from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import orjson
import soundfile as sf

import quran_audio_data.pipeline as pipeline
from quran_audio_data.alignment.base import AlignmentOutput
from quran_audio_data.schema import AyahTiming, WordTiming


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_DATA = PROJECT_ROOT / "data" / "quran_text_uthmani_v1.json"


def _write_silence(path: Path) -> None:
    sf.write(path, np.zeros(16000, dtype=np.float32), 16000)


def _write_manifest(path: Path, audio_path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["audio_path", "reciter_id", "surah", "ayah", "source_url", "sha256", "language"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "audio_path": str(audio_path),
                "reciter_id": "qc_meta",
                "surah": "1",
                "ayah": "1",
                "source_url": "",
                "sha256": "",
                "language": "ar",
            }
        )


def test_pipeline_records_qc_audio_methods(tmp_path, monkeypatch) -> None:
    audio = tmp_path / "sample.wav"
    manifest = tmp_path / "manifest.csv"
    out_dir = tmp_path / "out"
    _write_silence(audio)
    _write_manifest(manifest, audio)

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

    monkeypatch.setattr(pipeline, "NemoAligner", FakeNemo)
    monkeypatch.setattr(pipeline, "WhisperXFallbackAligner", FakeWhisperX)
    monkeypatch.setattr(pipeline, "MFAAligner", FakeMFA)
    monkeypatch.setattr(pipeline, "_refine_word_boundaries", lambda **kwargs: kwargs["words"])

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

    payload = orjson.loads((out_dir / "qc_meta_s001_a001.json").read_bytes())
    assert payload["qc"]["speech_end_method"] in {"none", "numpy", "webrtcvad"}
    assert payload["qc"]["boundary_refine_method"] in {"none", "numpy", "librosa"}
