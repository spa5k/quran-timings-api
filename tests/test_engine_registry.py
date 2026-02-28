from __future__ import annotations

import pytest

from quran_audio_data.pipeline.engine_registry import EngineRegistry
from quran_audio_data.pipeline.types import PipelineError


class _AvailableEngine:
    def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):  # noqa: ANN001
        raise NotImplementedError


class _UnavailableEngine:
    def is_available(self) -> bool:
        return False

    def availability_error(self) -> str:
        return "missing dependency"

    def align(self, *, audio_wav_path, canonical_words, audio_duration_s, device):  # noqa: ANN001
        raise NotImplementedError


def test_engine_registry_best_effort_skips_unavailable_requested_engine() -> None:
    registry = EngineRegistry(
        nemo=_UnavailableEngine(),
        whisperx=_AvailableEngine(),
        mfa=_AvailableEngine(),
    )

    selection = registry.select(
        requested_engine="nemo",
        multi_engine=["nemo", "whisperx"],
        policy="best_effort",
    )

    assert selection.requested_engine == "whisperx"
    assert selection.engines_to_try == ["whisperx", "mfa"]
    assert selection.unavailable["nemo"] == "missing dependency"


def test_engine_registry_require_requested_raises_for_unavailable_requested() -> None:
    registry = EngineRegistry(
        nemo=_UnavailableEngine(),
        whisperx=_AvailableEngine(),
        mfa=_AvailableEngine(),
    )

    with pytest.raises(PipelineError):
        registry.select(
            requested_engine="nemo",
            multi_engine=["nemo", "whisperx"],
            policy="require_requested",
        )


def test_engine_registry_require_all_raises_when_any_unavailable() -> None:
    registry = EngineRegistry(
        nemo=_AvailableEngine(),
        whisperx=_UnavailableEngine(),
        mfa=_AvailableEngine(),
    )

    with pytest.raises(PipelineError):
        registry.select(
            requested_engine="nemo",
            multi_engine=["nemo", "whisperx", "mfa"],
            policy="require_all",
        )
