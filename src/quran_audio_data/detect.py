from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import unicodedata
from urllib.parse import urlparse

import numpy as np
from rapidfuzz import fuzz
import soundfile as sf

from quran_audio_data.alignment.nemo_runner import load_nemo_model
from quran_audio_data.pipeline.audio import ensure_wav_16k_mono
from quran_audio_data.text.quran_text import QuranTextStore, normalize_arabic


@dataclass(slots=True)
class UrlReferenceHint:
    surah: int | None
    ayah: int | None
    scope: str


@dataclass(slots=True)
class DetectionCandidate:
    surah: int
    ayah: int
    score: float
    ayah_end: int | None = None


@dataclass(slots=True)
class DetectionResult:
    surah: int
    ayah: int
    score: float
    backend: str
    transcript: str
    transcript_norm: str
    candidates: list[DetectionCandidate]
    ayah_end: int | None = None


@dataclass(slots=True)
class VerseReference:
    surah: int
    ayah: int
    text_norm: str


_BISMILLAH_NORM = normalize_arabic("بسم الله الرحمن الرحيم")


def parse_reference_hint_from_audio_url(url: str) -> UrlReferenceHint:
    path = urlparse(url).path
    stem = re.sub(r"\.[A-Za-z0-9]+$", "", path.split("/")[-1].strip())
    if not stem:
        return UrlReferenceHint(surah=None, ayah=None, scope="unknown")

    m_full = re.fullmatch(r"(?P<surah>\d{3})(?P<ayah>\d{3})", stem)
    if m_full is not None:
        surah = int(m_full.group("surah"))
        ayah = int(m_full.group("ayah"))
        if 1 <= surah <= 114 and ayah >= 1:
            return UrlReferenceHint(surah=surah, ayah=ayah, scope="ayah_file")

    m_sep = re.fullmatch(r"(?P<surah>\d{1,3})[-_](?P<ayah>\d{1,3})", stem)
    if m_sep is not None:
        surah = int(m_sep.group("surah"))
        ayah = int(m_sep.group("ayah"))
        if 1 <= surah <= 114 and ayah >= 1:
            return UrlReferenceHint(surah=surah, ayah=ayah, scope="ayah_file")

    m_surah = re.fullmatch(r"(?P<surah>\d{3})", stem)
    if m_surah is not None:
        surah = int(m_surah.group("surah"))
        if 1 <= surah <= 114:
            return UrlReferenceHint(surah=surah, ayah=None, scope="surah_file")

    return UrlReferenceHint(surah=None, ayah=None, scope="unknown")


def infer_reciter_name_from_audio_url(url: str) -> str | None:
    segments = [piece for piece in urlparse(url).path.split("/") if piece]
    if not segments:
        return None

    for idx, segment in enumerate(segments):
        if segment.lower() == "quran" and idx + 1 < len(segments):
            raw = segments[idx + 1]
            break
    else:
        raw = ""

    if not raw:
        return None

    name = raw.replace("_", " ").replace("-", " ").strip()
    return name or None


def slugify_reciter_name(name: str) -> str:
    value = unicodedata.normalize("NFKD", name).lower()
    value = "".join(ch for ch in value if not unicodedata.combining(ch))
    value = value.replace("/", " ")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "reciter"


def detect_surah_ayah(
    *,
    audio_path: str,
    text_store: QuranTextStore,
    model_name: str,
    surah_hint: int | None = None,
    top_k: int = 5,
    backend: str = "auto",
    onnx_model_path: str | Path | None = None,
    onnx_vocab_path: str | Path | None = None,
    onnx_quran_path: str | Path | None = None,
) -> DetectionResult:
    backend_norm = backend.strip().lower()
    if backend_norm not in {"auto", "onnx", "nemo"}:
        raise ValueError(f"unsupported detect backend: {backend}")

    onnx_ready = False
    if onnx_model_path is not None and onnx_vocab_path is not None:
        onnx_ready = Path(onnx_model_path).exists() and Path(onnx_vocab_path).exists()

    if backend_norm == "onnx" and not onnx_ready:
        raise RuntimeError(
            "ONNX detection requested, but model/vocab paths are missing. "
            "Provide --onnx-model and --onnx-vocab."
        )

    if backend_norm in {"auto", "onnx"} and onnx_ready:
        try:
            return _detect_surah_ayah_with_onnx(
                audio_path=audio_path,
                text_store=text_store,
                surah_hint=surah_hint,
                top_k=top_k,
                onnx_model_path=Path(onnx_model_path),
                onnx_vocab_path=Path(onnx_vocab_path),
                onnx_quran_path=(
                    Path(onnx_quran_path)
                    if onnx_quran_path is not None and str(onnx_quran_path).strip()
                    else None
                ),
            )
        except Exception:
            if backend_norm == "onnx":
                raise

    return _detect_surah_ayah_with_nemo(
        audio_path=audio_path,
        text_store=text_store,
        model_name=model_name,
        surah_hint=surah_hint,
        top_k=top_k,
    )


def _detect_surah_ayah_with_nemo(
    *,
    audio_path: str,
    text_store: QuranTextStore,
    model_name: str,
    surah_hint: int | None = None,
    top_k: int = 5,
) -> DetectionResult:
    transcript = _transcribe_with_nemo(audio_path=audio_path, model_name=model_name)
    transcript_norm = normalize_arabic(transcript)
    if not transcript_norm:
        raise RuntimeError("empty transcript produced during detection")

    candidates = rank_ayah_candidates(
        transcript_norm=transcript_norm,
        text_store=text_store,
        surah_hint=surah_hint,
        top_k=top_k,
    )
    if not candidates:
        raise RuntimeError("unable to find candidate ayahs for detected transcript")

    top = candidates[0]
    return DetectionResult(
        surah=top.surah,
        ayah=top.ayah,
        score=top.score,
        backend="nemo",
        transcript=transcript,
        transcript_norm=transcript_norm,
        candidates=candidates,
        ayah_end=top.ayah_end,
    )


def rank_ayah_candidates(
    *,
    transcript_norm: str,
    text_store: QuranTextStore,
    surah_hint: int | None = None,
    top_k: int = 5,
) -> list[DetectionCandidate]:
    references = _references_from_text_store(text_store=text_store)
    return rank_ayah_candidates_from_references(
        transcript_norm=transcript_norm,
        references=references,
        surah_hint=surah_hint,
        top_k=top_k,
    )


def rank_ayah_candidates_from_references(
    *,
    transcript_norm: str,
    references: list[VerseReference],
    surah_hint: int | None = None,
    top_k: int = 5,
    max_span: int = 4,
) -> list[DetectionCandidate]:
    if top_k < 1:
        top_k = 1
    if max_span < 1:
        max_span = 1
    query = normalize_arabic(transcript_norm)
    if not query:
        return []

    singles: list[DetectionCandidate] = []
    by_surah: dict[int, list[VerseReference]] = {}
    for ref in references:
        by_surah.setdefault(ref.surah, []).append(ref)
        if surah_hint is not None and ref.surah != surah_hint:
            continue
        score = _fuzzy_match_score(query=query, candidate=ref.text_norm)
        stripped = _strip_bismillah_prefix(ref)
        if stripped:
            score = max(score, _fuzzy_match_score(query=query, candidate=stripped))
        singles.append(DetectionCandidate(surah=ref.surah, ayah=ref.ayah, score=score))

    singles.sort(key=lambda item: (-item.score, item.surah, item.ayah))
    if not singles:
        return []

    if surah_hint is not None:
        candidate_surahs = [surah_hint]
    else:
        candidate_surahs = []
        seen_surahs: set[int] = set()
        for item in singles[:80]:
            if item.surah in seen_surahs:
                continue
            seen_surahs.add(item.surah)
            candidate_surahs.append(item.surah)
            if len(candidate_surahs) >= 24:
                break

    scored: list[DetectionCandidate] = list(singles[: max(80, top_k * 8)])
    if max_span > 1:
        for surah in candidate_surahs:
            ayahs = by_surah.get(surah, [])
            if not ayahs:
                continue
            for i in range(len(ayahs)):
                upper = min(len(ayahs), i + max_span)
                for j in range(i + 1, upper):
                    chunk = ayahs[i : j + 1]
                    start = chunk[0]
                    combined = _join_reference_span(chunk)
                    score = _fuzzy_match_score(query=query, candidate=combined)
                    scored.append(
                        DetectionCandidate(
                            surah=surah,
                            ayah=start.ayah,
                            ayah_end=chunk[-1].ayah,
                            score=score,
                        )
                    )

    deduped: dict[tuple[int, int, int | None], DetectionCandidate] = {}
    for item in scored:
        key = (item.surah, item.ayah, item.ayah_end)
        existing = deduped.get(key)
        if existing is None or item.score > existing.score:
            deduped[key] = item

    ranked = sorted(
        deduped.values(),
        key=lambda item: (-item.score, item.surah, item.ayah, item.ayah_end or item.ayah),
    )
    return ranked[:top_k]


def _join_reference_span(span: list[VerseReference]) -> str:
    if not span:
        return ""
    pieces: list[str] = []
    for idx, ref in enumerate(span):
        text = ref.text_norm
        if idx == 0:
            stripped = _strip_bismillah_prefix(ref)
            if stripped:
                text = stripped
        if text:
            pieces.append(text)
    return " ".join(pieces).strip()


def _strip_bismillah_prefix(ref: VerseReference) -> str | None:
    if ref.ayah != 1 or ref.surah in {1, 9}:
        return None
    if not ref.text_norm.startswith(_BISMILLAH_NORM):
        return None
    stripped = ref.text_norm[len(_BISMILLAH_NORM) :].strip()
    return stripped or None


def _fuzzy_match_score(*, query: str, candidate: str) -> float:
    if not query or not candidate:
        return 0.0
    ratio = float(fuzz.ratio(query, candidate))
    partial = float(fuzz.partial_ratio(query, candidate))
    token_set = float(fuzz.token_set_ratio(query, candidate))
    base = (0.50 * partial) + (0.35 * ratio) + (0.15 * token_set)

    q_len = max(len(query), 1)
    c_len = max(len(candidate), 1)
    length_ratio = min(q_len, c_len) / max(q_len, c_len)
    if q_len <= 8:
        # Very short decodes are noisy; discourage overconfident false matches.
        length_weight = 0.35 + (0.65 * length_ratio)
    else:
        length_weight = 0.6 + (0.4 * length_ratio)
    return base * length_weight


def load_quran_references(
    *,
    text_store: QuranTextStore,
    quran_path: str | Path | None,
) -> list[VerseReference]:
    if quran_path is not None:
        path = Path(quran_path)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            parsed = _parse_quran_payload(payload)
            if parsed:
                return parsed
    return _references_from_text_store(text_store=text_store)


def decode_ctc_greedy(
    *,
    logits: np.ndarray,
    id_to_token: dict[int, str],
) -> str:
    if logits.ndim != 2:
        raise ValueError(f"logits must be rank-2 [T, V], got shape {logits.shape}")

    blank_id = _infer_blank_id(id_to_token)
    ids = np.argmax(logits, axis=1)
    previous = -1
    tokens: list[str] = []
    for current in ids:
        token_id = int(current)
        if token_id != previous and token_id != blank_id:
            tokens.append(id_to_token.get(token_id, ""))
        previous = token_id

    transcript = "".join(tokens)
    transcript = transcript.replace("▁", " ").replace("\u2581", " ")
    transcript = re.sub(r"\s+", " ", transcript).strip()
    return transcript


def _detect_surah_ayah_with_onnx(
    *,
    audio_path: str,
    text_store: QuranTextStore,
    surah_hint: int | None,
    top_k: int,
    onnx_model_path: Path,
    onnx_vocab_path: Path,
    onnx_quran_path: Path | None,
) -> DetectionResult:
    references = load_quran_references(
        text_store=text_store,
        quran_path=onnx_quran_path,
    )
    session, id_to_token = _load_onnx_session_and_vocab(
        onnx_model_path=onnx_model_path,
        onnx_vocab_path=onnx_vocab_path,
    )
    audio_f32 = _load_audio_16k_mono_for_onnx(audio_path)
    transcript = _decode_onnx_audio(
        audio_f32=audio_f32,
        session=session,
        id_to_token=id_to_token,
    )
    transcript_norm = normalize_arabic(transcript)
    if not transcript_norm:
        raise RuntimeError("empty transcript produced during ONNX detection")

    candidates = rank_ayah_candidates_from_references(
        transcript_norm=transcript_norm,
        references=references,
        surah_hint=surah_hint,
        top_k=top_k,
    )
    if not candidates:
        raise RuntimeError("unable to find candidate ayahs for ONNX transcript")

    backend_name = "onnx"
    primary_top = candidates[0]
    if len(transcript_norm) < 12 or primary_top.score < 60.0:
        window_candidates, window_transcript = _windowed_onnx_candidates(
            audio_f32=audio_f32,
            session=session,
            id_to_token=id_to_token,
            references=references,
            surah_hint=surah_hint,
            top_k=top_k,
        )
        if window_candidates and window_candidates[0].score >= primary_top.score:
            candidates = window_candidates
            transcript = window_transcript or transcript
            transcript_norm = normalize_arabic(transcript)
            backend_name = "onnx-windowed"

    top = candidates[0]
    return DetectionResult(
        surah=top.surah,
        ayah=top.ayah,
        score=top.score,
        backend=backend_name,
        transcript=transcript,
        transcript_norm=transcript_norm,
        candidates=candidates,
        ayah_end=top.ayah_end,
    )


def _transcribe_with_onnx(
    *,
    audio_path: str,
    onnx_model_path: Path,
    onnx_vocab_path: Path,
) -> str:
    session, id_to_token = _load_onnx_session_and_vocab(
        onnx_model_path=onnx_model_path,
        onnx_vocab_path=onnx_vocab_path,
    )
    audio_f32 = _load_audio_16k_mono_for_onnx(audio_path)
    transcript = _decode_onnx_audio(
        audio_f32=audio_f32,
        session=session,
        id_to_token=id_to_token,
    )
    if not transcript:
        raise RuntimeError("ONNX CTC decode produced empty transcript")
    return transcript


def _load_onnx_session_and_vocab(
    *,
    onnx_model_path: Path,
    onnx_vocab_path: Path,
):
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError(
            "onnxruntime is required for Offline Tarteel detection. "
            "Install it (for example: uv add onnxruntime or uv sync --extra detect)."
        ) from exc
    session = ort.InferenceSession(str(onnx_model_path))
    id_to_token = _load_vocab(onnx_vocab_path)
    return session, id_to_token


def _load_audio_16k_mono_for_onnx(audio_path: str) -> np.ndarray:
    wav_path = ensure_wav_16k_mono(audio_path)
    audio, sample_rate = sf.read(str(wav_path))
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)
    audio_f32 = np.asarray(audio, dtype=np.float32)
    if sample_rate != 16000:
        raise RuntimeError(f"expected 16kHz WAV after preprocessing, got {sample_rate}")
    if audio_f32.size == 0:
        raise RuntimeError("audio is empty after preprocessing")
    return audio_f32


def _decode_onnx_audio(
    *,
    audio_f32: np.ndarray,
    session,
    id_to_token: dict[int, str],
) -> str:
    features = _compute_nemo_mel_features(audio_f32)
    time_frames = int(features.shape[1])
    if time_frames <= 0:
        return ""
    inputs = session.get_inputs()
    if len(inputs) < 2:
        raise RuntimeError(
            f"expected at least 2 ONNX inputs (features,length), got {len(inputs)}"
        )

    features_batched = features.astype(np.float32, copy=False)[np.newaxis, :, :]
    length = np.array([time_frames], dtype=np.int64)
    outputs = session.run(
        None,
        {
            inputs[0].name: features_batched,
            inputs[1].name: length,
        },
    )
    if not outputs:
        return ""

    raw = np.asarray(outputs[0])
    logits = _to_time_major_logits(raw, expected_time_frames=time_frames)
    return decode_ctc_greedy(logits=logits, id_to_token=id_to_token)


def _windowed_onnx_candidates(
    *,
    audio_f32: np.ndarray,
    session,
    id_to_token: dict[int, str],
    references: list[VerseReference],
    surah_hint: int | None,
    top_k: int,
) -> tuple[list[DetectionCandidate], str]:
    sample_rate = 16000
    duration_s = float(audio_f32.size) / float(sample_rate)
    if duration_s < 4.0:
        return [], ""

    if duration_s <= 12.0:
        window_s = max(4.0, duration_s)
    elif duration_s <= 45.0:
        window_s = 8.0
    else:
        window_s = 12.0
    stride_s = max(3.0, window_s / 2.0)

    windows: list[tuple[float, float]] = []
    start = 0.0
    while start < duration_s:
        end = min(duration_s, start + window_s)
        if end - start >= 3.5:
            windows.append((start, end))
        if end >= duration_s:
            break
        start += stride_s

    if not windows:
        return [], ""

    aggregate: dict[tuple[int, int, int | None], float] = {}
    surah_votes: dict[int, float] = {}
    snippets: list[str] = []
    used_windows = 0

    for start_s, end_s in windows:
        start_idx = int(start_s * sample_rate)
        end_idx = int(end_s * sample_rate)
        chunk = audio_f32[start_idx:end_idx]
        if chunk.size < int(3.5 * sample_rate):
            continue

        transcript = _decode_onnx_audio(
            audio_f32=chunk,
            session=session,
            id_to_token=id_to_token,
        )
        transcript_norm = normalize_arabic(transcript)
        if len(transcript_norm) < 2:
            continue
        snippets.append(transcript_norm)

        local = rank_ayah_candidates_from_references(
            transcript_norm=transcript_norm,
            references=references,
            surah_hint=surah_hint,
            top_k=max(3, min(8, top_k + 2)),
        )
        if not local:
            continue
        used_windows += 1
        for rank, candidate in enumerate(local):
            weight = candidate.score / float(rank + 1)
            key = (candidate.surah, candidate.ayah, candidate.ayah_end)
            aggregate[key] = aggregate.get(key, 0.0) + weight
            surah_votes[candidate.surah] = surah_votes.get(candidate.surah, 0.0) + weight

    if not aggregate:
        return [], ""

    target_surah = surah_hint
    if target_surah is None and surah_votes:
        target_surah = max(surah_votes.items(), key=lambda item: item[1])[0]

    scored: list[DetectionCandidate] = []
    for (surah, ayah, ayah_end), value in aggregate.items():
        if target_surah is not None and surah != target_surah:
            continue
        scored.append(
            DetectionCandidate(
                surah=surah,
                ayah=ayah,
                ayah_end=ayah_end,
                score=value,
            )
        )
    if not scored:
        return [], ""

    scored.sort(key=lambda item: (-item.score, item.surah, item.ayah, item.ayah_end or item.ayah))
    norm = max(float(used_windows), 1.0)
    normalized = [
        DetectionCandidate(
            surah=item.surah,
            ayah=item.ayah,
            ayah_end=item.ayah_end,
            score=min(100.0, item.score / norm),
        )
        for item in scored[:top_k]
    ]
    return normalized, " | ".join(snippets[:4])


def _to_time_major_logits(raw: np.ndarray, *, expected_time_frames: int) -> np.ndarray:
    if raw.ndim == 3:
        matrix = raw[0]
    elif raw.ndim == 2:
        matrix = raw
    else:
        raise RuntimeError(f"unexpected ONNX output shape: {raw.shape}")

    if matrix.ndim != 2:
        raise RuntimeError(f"unexpected ONNX matrix rank: {matrix.ndim}")

    if matrix.shape[0] == expected_time_frames:
        return matrix
    if matrix.shape[1] == expected_time_frames:
        return matrix.T

    if matrix.shape[0] < matrix.shape[1]:
        return matrix
    return matrix.T


def _load_vocab(path: Path) -> dict[int, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: dict[int, str] = {}
    if isinstance(payload, list):
        for idx, token in enumerate(payload):
            out[idx] = _token_text(token)
        return out

    if isinstance(payload, dict):
        for key, token in payload.items():
            try:
                idx = int(key)
            except Exception:
                continue
            out[idx] = _token_text(token)
    if not out:
        raise RuntimeError(f"failed to parse vocab mapping from {path}")
    return out


def _token_text(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("token", "piece", "text", "value", "char"):
            candidate = value.get(key)
            if candidate is not None:
                return str(candidate)
    return ""


def _infer_blank_id(id_to_token: dict[int, str]) -> int:
    for idx, token in id_to_token.items():
        if token in {"<blank>", "<BLANK>", "<blk>", "<pad>"}:
            return idx
    return max(id_to_token)


def _compute_nemo_mel_features(audio: np.ndarray) -> np.ndarray:
    try:
        import librosa
    except Exception as exc:
        raise RuntimeError(
            "librosa is required for ONNX mel feature extraction "
            "(install with: uv sync --extra audio)."
        ) from exc

    dither = np.random.default_rng(0).standard_normal(audio.shape).astype(np.float32)
    audio = audio + (1e-5 * dither)
    emphasized = np.empty_like(audio, dtype=np.float32)
    emphasized[0] = audio[0]
    emphasized[1:] = audio[1:] - (0.97 * audio[:-1])

    mel = librosa.feature.melspectrogram(
        y=emphasized,
        sr=16000,
        n_fft=512,
        hop_length=160,
        win_length=400,
        n_mels=80,
        fmax=8000,
        htk=True,
        norm="slaney",
        power=2.0,
        center=False,
    )
    mel = np.log(mel + 1e-5)
    mel = (mel - mel.mean(axis=1, keepdims=True)) / (
        mel.std(axis=1, keepdims=True) + 1e-10
    )
    return mel.astype(np.float32, copy=False)


def _references_from_text_store(*, text_store: QuranTextStore) -> list[VerseReference]:
    refs: list[VerseReference] = []
    for surah in range(1, 115):
        try:
            ayah_rows = text_store.get_surah_ayahs(surah)
        except KeyError:
            continue
        for ayah, text in ayah_rows:
            text_norm = normalize_arabic(text)
            if text_norm:
                refs.append(VerseReference(surah=surah, ayah=ayah, text_norm=text_norm))
    return refs


def _parse_quran_payload(payload: object) -> list[VerseReference]:
    refs: list[VerseReference] = []

    if isinstance(payload, dict):
        surahs = payload.get("surahs")
        if isinstance(surahs, dict):
            for surah_key, ayah_map in surahs.items():
                try:
                    surah = int(surah_key)
                except Exception:
                    continue
                if not isinstance(ayah_map, dict):
                    continue
                for ayah_key, text in ayah_map.items():
                    try:
                        ayah = int(ayah_key)
                    except Exception:
                        continue
                    text_norm = normalize_arabic(str(text or ""))
                    if text_norm:
                        refs.append(
                            VerseReference(surah=surah, ayah=ayah, text_norm=text_norm)
                        )

        for key, value in payload.items():
            if ":" not in str(key):
                continue
            parts = str(key).split(":", 1)
            try:
                surah = int(parts[0])
                ayah = int(parts[1])
            except Exception:
                continue
            text_norm = normalize_arabic(str(value or ""))
            if text_norm:
                refs.append(VerseReference(surah=surah, ayah=ayah, text_norm=text_norm))

    elif isinstance(payload, list):
        for row in payload:
            if not isinstance(row, dict):
                continue
            surah = _coerce_int(
                row.get("surah")
                or row.get("sura")
                or row.get("chapter")
                or row.get("chapter_id")
            )
            ayah = _coerce_int(
                row.get("ayah")
                or row.get("verse")
                or row.get("ayah_number")
                or row.get("verse_number")
                or row.get("number_in_surah")
            )
            text = (
                row.get("text")
                or row.get("text_clean")
                or row.get("ayah_text")
                or row.get("verse_text")
                or row.get("text_uthmani")
                or row.get("uthmani")
                or row.get("value")
            )
            if surah is None or ayah is None:
                continue
            text_norm = normalize_arabic(str(text or ""))
            if text_norm:
                refs.append(VerseReference(surah=surah, ayah=ayah, text_norm=text_norm))

    deduped = {(item.surah, item.ayah): item for item in refs}
    return sorted(deduped.values(), key=lambda item: (item.surah, item.ayah))


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _transcribe_with_nemo(*, audio_path: str, model_name: str) -> str:
    wav_path = ensure_wav_16k_mono(audio_path)
    model, torch, _resolved_device = load_nemo_model(model_name=model_name, device="auto")
    with torch.no_grad():
        hypotheses = model.transcribe(
            [str(wav_path)],
            batch_size=1,
            verbose=False,
            return_hypotheses=False,
        )

    if isinstance(hypotheses, tuple) and hypotheses:
        hypotheses = hypotheses[0]
    if not isinstance(hypotheses, list) or not hypotheses:
        raise RuntimeError("NeMo transcribe returned no hypotheses")

    sample = hypotheses[0]
    if isinstance(sample, str):
        transcript = sample
    else:
        transcript = str(getattr(sample, "text", "") or "")

    transcript = transcript.strip()
    if not transcript:
        raise RuntimeError("NeMo transcribe returned an empty transcript")
    return transcript


__all__ = [
    "DetectionCandidate",
    "DetectionResult",
    "UrlReferenceHint",
    "decode_ctc_greedy",
    "detect_surah_ayah",
    "infer_reciter_name_from_audio_url",
    "load_quran_references",
    "parse_reference_hint_from_audio_url",
    "rank_ayah_candidates",
    "rank_ayah_candidates_from_references",
    "slugify_reciter_name",
]
