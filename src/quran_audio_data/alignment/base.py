from __future__ import annotations

from dataclasses import dataclass

from quran_audio_data.schema import AyahTiming, WordTiming


class AlignmentError(RuntimeError):
    pass


class EngineUnavailable(AlignmentError):
    pass


@dataclass(slots=True)
class AlignmentOutput:
    ayahs: list[AyahTiming]
    words: list[WordTiming]
    engine_name: str
    engine_model: str
    device: str
    source: str
