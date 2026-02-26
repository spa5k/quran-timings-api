from __future__ import annotations

from quran_audio_data.alignment.nemo_aligner import NemoAligner
from quran_audio_data.alignment.nemo_runner import PredictedWord, map_reference_words


def test_nemo_aligner_uses_built_in_runner_by_default(monkeypatch) -> None:
    monkeypatch.delenv("QAD_NEMO_ALIGN_CMD", raising=False)
    aligner = NemoAligner()
    assert "-m quran_audio_data.alignment.nemo_runner" in aligner.command_template


def test_runner_mapping_preserves_transcript_length_and_monotonicity() -> None:
    transcript = ["بسم", "الله", "الرحمن", "الرحيم"]
    predicted = [
        PredictedWord(text_norm="بسم", start_s=0.0, end_s=0.4, confidence=0.9),
        PredictedWord(text_norm="الله", start_s=0.4, end_s=0.8, confidence=0.9),
        PredictedWord(text_norm="الرحيم", start_s=1.2, end_s=1.8, confidence=0.9),
    ]

    mapped = map_reference_words(
        transcript_words=transcript,
        predicted_words=predicted,
        audio_duration_s=2.0,
    )

    assert len(mapped) == len(transcript)
    starts = [row["start"] for row in mapped]
    assert starts == sorted(starts)
    assert mapped[-1]["end"] <= 2.0 + 1e-6
