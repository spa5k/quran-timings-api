from __future__ import annotations

from quran_audio_data.alignment import AlignmentError, EngineUnavailable
from quran_audio_data.alignment import MFAAligner as _MFAAligner
from quran_audio_data.alignment import NemoAligner as _NemoAligner
from quran_audio_data.alignment import WhisperXFallbackAligner as _WhisperXFallbackAligner

from .artifacts import validate_outputs
from .artifacts import write_cache_result as _write_cache_result
from .audio import ensure_wav_16k_mono, probe_audio, refine_word_boundaries as _default_refine_word_boundaries, sha256_file
from .engine_registry import EngineProtocol, EngineRegistry
from .manifest import ManifestRow, read_manifest
from .orchestrator import _normalize_engines, benchmark_pipeline as _benchmark_pipeline
from .orchestrator import run_alignment_pipeline as _run_alignment_pipeline
from .types import (
    AccuracyMode,
    DeviceOption,
    EngineAvailabilityPolicy,
    EngineOption,
    PipelineError,
    PipelineReportV3,
    ProcessingSummary,
)


NemoAligner = _NemoAligner
WhisperXFallbackAligner = _WhisperXFallbackAligner
MFAAligner = _MFAAligner
_refine_word_boundaries = _default_refine_word_boundaries


def _init_nemo(model_name: str):
    try:
        return NemoAligner(model_name=model_name)
    except TypeError:
        return NemoAligner()


def _build_registry() -> EngineRegistry:
    return EngineRegistry(
        nemo=_init_nemo("nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"),
        whisperx=WhisperXFallbackAligner(),
        mfa=_init_nemo("nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0"),
    )


def run_alignment_pipeline(
    *,
    manifest_path,
    out_dir,
    engine: EngineOption = "nemo",
    multi_engine: list[EngineOption] | None = None,
    accuracy_mode: AccuracyMode = "strict",
    device: DeviceOption = "auto",
    text_data=None,
    cache_dir=None,
    enable_remote: bool = True,
    sample_size: int | None = None,
    thresholds=None,
    availability_policy: EngineAvailabilityPolicy = "best_effort",
) -> PipelineReportV3:
    return _run_alignment_pipeline(
        manifest_path=manifest_path,
        out_dir=out_dir,
        engine=engine,
        multi_engine=multi_engine,
        accuracy_mode=accuracy_mode,
        device=device,
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=enable_remote,
        sample_size=sample_size,
        thresholds=thresholds,
        availability_policy=availability_policy,
        registry=_build_registry(),
        refine_word_boundaries_fn=_refine_word_boundaries,
    )


def benchmark_pipeline(
    *,
    manifest_path,
    out_dir,
    sample_size: int,
    engine: EngineOption = "nemo",
    multi_engine: list[EngineOption] | None = None,
    accuracy_mode: AccuracyMode = "strict",
    device: DeviceOption = "auto",
    text_data=None,
    cache_dir=None,
    enable_remote: bool = True,
    availability_policy: EngineAvailabilityPolicy = "best_effort",
):
    return _benchmark_pipeline(
        manifest_path=manifest_path,
        out_dir=out_dir,
        sample_size=sample_size,
        engine=engine,
        multi_engine=multi_engine,
        accuracy_mode=accuracy_mode,
        device=device,
        text_data=text_data,
        cache_dir=cache_dir,
        enable_remote=enable_remote,
        availability_policy=availability_policy,
    )


__all__ = [
    "AccuracyMode",
    "DeviceOption",
    "EngineOption",
    "EngineAvailabilityPolicy",
    "EngineProtocol",
    "EngineRegistry",
    "AlignmentError",
    "EngineUnavailable",
    "ManifestRow",
    "PipelineError",
    "PipelineReportV3",
    "ProcessingSummary",
    "read_manifest",
    "run_alignment_pipeline",
    "benchmark_pipeline",
    "validate_outputs",
    "probe_audio",
    "ensure_wav_16k_mono",
    "sha256_file",
    "_normalize_engines",
    "_write_cache_result",
    "_refine_word_boundaries",
    "NemoAligner",
    "WhisperXFallbackAligner",
    "MFAAligner",
]
