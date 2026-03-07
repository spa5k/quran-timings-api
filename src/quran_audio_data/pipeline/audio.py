from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile
from typing import Callable

import numpy as np
import soundfile as sf

from quran_audio_data.core.settings import get_settings
from quran_audio_data.schema import AudioMetadata, WordTiming


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probe_audio(path: str | Path) -> AudioMetadata:
    from .types import PipelineError

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
    from .types import PipelineError

    src = Path(path)

    try:
        info = sf.info(str(src))
        if src.suffix.lower() == ".wav" and info.samplerate == 16000 and info.channels == 1:
            return src
    except RuntimeError:
        pass

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise PipelineError("ffmpeg is required for converting input audio to mono WAV 16k.")

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


def refine_word_boundaries(
    *,
    words: list[WordTiming],
    wav_path: Path,
    max_shift_s: float,
    min_duration_s: float,
) -> tuple[list[WordTiming], str]:
    if not words:
        return words, "none"

    try:
        audio, sample_rate = sf.read(str(wav_path))
    except Exception:
        return words, "none"

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not isinstance(audio, np.ndarray):
        return words, "none"
    if audio.size <= 1 or sample_rate <= 0:
        return words, "none"

    audio_f32 = audio.astype(np.float32, copy=False)
    settings = get_settings()

    if settings.use_librosa:
        try:
            import librosa

            hop = max(1, int(sample_rate * 0.005))
            frame = max(hop * 2, int(sample_rate * 0.02))
            rms = librosa.feature.rms(y=audio_f32, frame_length=frame, hop_length=hop)[0]
            frame_times = librosa.frames_to_time(
                np.arange(len(rms)), sr=sample_rate, hop_length=hop
            )
            max_shift = max(max_shift_s, 0.01)

            def snap(ts: float) -> float:
                left = max(0, int(np.searchsorted(frame_times, ts - max_shift, side="left")))
                right = min(
                    len(frame_times),
                    int(np.searchsorted(frame_times, ts + max_shift, side="right")),
                )
                if right <= left:
                    return ts
                region = rms[left:right]
                if region.size == 0:
                    return ts
                offset = int(np.argmin(region))
                snapped = float(frame_times[left + offset])
                return max(0.0, snapped)

            return _finalize_refined_words(
                words=words, snap=snap, min_duration_s=min_duration_s
            ), "librosa"
        except Exception:
            pass

    envelope = np.abs(audio_f32)
    if float(np.max(envelope)) < 1e-6:
        return words, "none"
    window = max(1, int(sample_rate * 0.005))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(envelope, kernel, mode="same")

    max_shift = max(1, int(sample_rate * max_shift_s))

    def snap(ts: float) -> float:
        center = int(max(0, min(len(smoothed) - 1, round(ts * sample_rate))))
        left = max(0, center - max_shift)
        right = min(len(smoothed), center + max_shift + 1)
        region = smoothed[left:right]
        if region.size == 0:
            return ts
        offset = int(np.argmin(region))
        return (left + offset) / sample_rate

    return _finalize_refined_words(words=words, snap=snap, min_duration_s=min_duration_s), "numpy"


def _finalize_refined_words(
    *,
    words: list[WordTiming],
    snap: Callable[[float], float],
    min_duration_s: float,
) -> list[WordTiming]:
    refined: list[WordTiming] = []
    min_duration = max(0.001, min_duration_s)
    previous_start = 0.0
    previous_end = 0.0
    for word in words:
        start_s = snap(word.start_s)
        end_s = snap(word.end_s)
        start_s = max(previous_start, start_s)
        if end_s < start_s + min_duration:
            end_s = start_s + min_duration
        if start_s < previous_end:
            start_s = previous_end
            if end_s < start_s + min_duration:
                end_s = start_s + min_duration
        previous_start = start_s
        previous_end = end_s
        refined.append(word.model_copy(update={"start_s": start_s, "end_s": end_s}))
    return refined


def estimate_speech_end_s(
    wav_path: Path,
    *,
    fallback_duration_s: float,
) -> tuple[float, str]:
    try:
        audio, sample_rate = sf.read(str(wav_path))
    except Exception:
        return fallback_duration_s, "none"

    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    if not isinstance(audio, np.ndarray):
        return fallback_duration_s, "none"
    if audio.size <= 1 or sample_rate <= 0:
        return fallback_duration_s, "none"

    settings = get_settings()
    if settings.use_webrtcvad:
        try:
            import webrtcvad

            vad = webrtcvad.Vad(2)
            target_rate = 16000
            if sample_rate == target_rate:
                pcm = np.clip(audio, -1.0, 1.0)
                pcm16 = (pcm * 32767.0).astype(np.int16)
                frame_ms = 30
                frame_size = int(target_rate * frame_ms / 1000)
                if frame_size > 0:
                    last_speech_end_s: float | None = None
                    total_frames = len(pcm16) // frame_size
                    for frame_idx in range(total_frames):
                        start = frame_idx * frame_size
                        end = start + frame_size
                        frame = pcm16[start:end].tobytes()
                        if vad.is_speech(frame, target_rate):
                            last_speech_end_s = end / float(target_rate)
                    if last_speech_end_s is not None:
                        return min(max(last_speech_end_s, 0.01), fallback_duration_s), "webrtcvad"
        except Exception:
            pass

    envelope = np.abs(audio.astype(np.float32, copy=False))
    peak = float(np.max(envelope))
    if peak <= 1e-6:
        return fallback_duration_s, "none"

    window = max(1, int(sample_rate * 0.02))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(envelope, kernel, mode="same")

    percentile = float(np.percentile(smoothed, 70))
    threshold = max(peak * 0.035, percentile * 0.5, 1e-4)
    active = np.where(smoothed >= threshold)[0]
    if active.size == 0:
        return fallback_duration_s, "none"

    end_index = int(active[-1])
    speech_end_s = end_index / float(sample_rate)
    if speech_end_s <= 0:
        return fallback_duration_s, "none"
    return min(max(speech_end_s, 0.01), fallback_duration_s), "numpy"


def cut_audio_chunk(
    *,
    ffmpeg: str,
    src_wav: Path,
    dst_wav: Path,
    start_s: float,
    end_s: float,
) -> None:
    from .types import PipelineError

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


__all__ = [
    "probe_audio",
    "ensure_wav_16k_mono",
    "sha256_file",
    "estimate_speech_end_s",
    "refine_word_boundaries",
    "cut_audio_chunk",
]
