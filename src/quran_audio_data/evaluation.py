from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import orjson

from quran_audio_data.supervision import is_qcom_word_supervision_supported


@dataclass(slots=True)
class ErrorMetrics:
    median_abs_error_ms: float
    p90_abs_error_ms: float
    p95_abs_error_ms: float
    hit_rate_20ms: float
    hit_rate_50ms: float
    hit_rate_80ms: float

    def to_dict(self) -> dict[str, float]:
        return {
            "median_abs_error_ms": self.median_abs_error_ms,
            "p90_abs_error_ms": self.p90_abs_error_ms,
            "p95_abs_error_ms": self.p95_abs_error_ms,
            "hit_rate_20ms": self.hit_rate_20ms,
            "hit_rate_50ms": self.hit_rate_50ms,
            "hit_rate_80ms": self.hit_rate_80ms,
        }


def _as_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        parsed = float(value)
        if not math.isfinite(parsed):
            return None
        return parsed
    except (TypeError, ValueError):
        return None


def _extract_key_and_words(
    payload: dict[str, Any], file_path: Path
) -> list[tuple[tuple[str, int, int], list[dict[str, Any]]]]:
    words = payload.get("words")
    if not isinstance(words, list) or not words:
        return []

    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    reciter_id = str(meta.get("reciter_id") or payload.get("reciter_id") or file_path.stem).strip()
    surah = _as_int(meta.get("surah") or payload.get("surah"))

    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for word in words:
        if not isinstance(word, dict):
            continue
        ayah = _as_int(word.get("ayah"))
        if ayah is None:
            continue
        grouped[ayah].append(word)
        if surah is None:
            surah = _as_int(word.get("surah"))

    if surah is None:
        return []

    out: list[tuple[tuple[str, int, int], list[dict[str, Any]]]] = []
    for ayah, ayah_words in grouped.items():
        ayah_words.sort(
            key=lambda item: (
                _as_int(item.get("word_index_in_ayah")) or 0,
                float(item.get("start_s") or 0.0),
            )
        )
        out.append(((reciter_id, surah, ayah), ayah_words))
    return out


def _collect_dataset(path: str | Path) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    root = Path(path)
    files = sorted(root.rglob("*.json")) if root.is_dir() else [root]

    dataset: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for file_path in files:
        if file_path.name.endswith("_qc_report.json"):
            continue
        try:
            payload = orjson.loads(file_path.read_bytes())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue

        for key, words in _extract_key_and_words(payload, file_path):
            dataset[key] = words

    return dataset


def _word_lookup(words: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for idx, word in enumerate(words, start=1):
        key = _as_int(word.get("word_index_in_ayah")) or idx
        out[key] = word
    return out


def _metrics_from_errors(boundary_errors_ms: list[float]) -> ErrorMetrics:
    if not boundary_errors_ms:
        return ErrorMetrics(
            median_abs_error_ms=0.0,
            p90_abs_error_ms=0.0,
            p95_abs_error_ms=0.0,
            hit_rate_20ms=0.0,
            hit_rate_50ms=0.0,
            hit_rate_80ms=0.0,
        )

    total = len(boundary_errors_ms)
    values = np.asarray(boundary_errors_ms, dtype=np.float64)
    return ErrorMetrics(
        median_abs_error_ms=float(median(boundary_errors_ms)),
        p90_abs_error_ms=float(np.percentile(values, 90)),
        p95_abs_error_ms=float(np.percentile(values, 95)),
        hit_rate_20ms=sum(1 for value in boundary_errors_ms if value <= 20.0) / total,
        hit_rate_50ms=sum(1 for value in boundary_errors_ms if value <= 50.0) / total,
        hit_rate_80ms=sum(1 for value in boundary_errors_ms if value <= 80.0) / total,
    )


def evaluate_predictions(
    *,
    pred_dir: str | Path,
    reference_dir: str | Path,
) -> dict[str, Any]:
    predictions = _collect_dataset(pred_dir)
    reference = _collect_dataset(reference_dir)

    boundary_errors_ms: list[float] = []
    per_reciter_errors: dict[str, list[float]] = defaultdict(list)
    missing_predictions: list[str] = []
    matched_ayahs = 0

    for key, reference_words in reference.items():
        reciter_id, surah, ayah = key
        pred_words = predictions.get(key)
        if pred_words is None:
            missing_predictions.append(f"{reciter_id}:{surah}:{ayah}")
            continue

        reference_lookup = _word_lookup(reference_words)
        pred_lookup = _word_lookup(pred_words)

        for word_idx, reference_word in reference_lookup.items():
            pred_word = pred_lookup.get(word_idx)
            if pred_word is None:
                continue
            try:
                reference_start = float(reference_word.get("start_s"))
                reference_end = float(reference_word.get("end_s"))
                pred_start = float(pred_word.get("start_s"))
                pred_end = float(pred_word.get("end_s"))
            except (TypeError, ValueError):
                continue

            start_err = abs(pred_start - reference_start) * 1000.0
            end_err = abs(pred_end - reference_end) * 1000.0
            boundary_errors_ms.extend([start_err, end_err])
            per_reciter_errors[reciter_id].extend([start_err, end_err])

        matched_ayahs += 1

    summary_metrics = _metrics_from_errors(boundary_errors_ms)
    per_reciter = {
        reciter_id: _metrics_from_errors(values).to_dict()
        for reciter_id, values in sorted(per_reciter_errors.items())
    }

    report = {
        "reference_ayahs": len(reference),
        "pred_ayahs": len(predictions),
        "matched_ayahs": matched_ayahs,
        "missing_predictions": missing_predictions,
        "summary": summary_metrics.to_dict(),
        "per_reciter": per_reciter,
        "targets_ms": {
            "median": 50.0,
            "p95": 120.0,
            "per_reciter_median": 65.0,
        },
        "passes_targets": (
            summary_metrics.median_abs_error_ms <= 50.0
            and summary_metrics.p95_abs_error_ms <= 120.0
            and all(metrics["median_abs_error_ms"] <= 65.0 for metrics in per_reciter.values())
        ),
    }

    return report


def _collect_prediction_metadata(path: str | Path) -> dict[str, Any]:
    root = Path(path)
    files = sorted(root.rglob("*.json")) if root.is_dir() else [root]

    reciters: set[str] = set()
    supervision_source_counter: dict[str, int] = defaultdict(int)
    segment_shape_counter: dict[str, int] = defaultdict(int)
    segment_source_type_counter: dict[str, int] = defaultdict(int)

    for file_path in files:
        if file_path.name.endswith("_qc_report.json") or file_path.name.endswith(
            "_text_audit.json"
        ):
            continue
        try:
            payload = orjson.loads(file_path.read_bytes())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        meta = payload.get("meta")
        if isinstance(meta, dict):
            reciter_id = str(meta.get("reciter_id") or "").strip()
            if reciter_id:
                reciters.add(reciter_id)

        for source in (
            payload.get("supervision_sources", [])
            if isinstance(payload.get("supervision_sources"), list)
            else []
        ):
            source_key = str(source)
            supervision_source_counter[source_key] += 1
            if "shape=3_field" in source_key:
                segment_shape_counter["3_field"] += 1
            elif "shape=4_field" in source_key:
                segment_shape_counter["4_field"] += 1

        segment_source_type = str(payload.get("segment_source_type") or "none")
        segment_source_type_counter[segment_source_type] += 1

    total_supervision_sources = sum(supervision_source_counter.values())
    supervision_source_usage_ratio = {
        key: (value / total_supervision_sources if total_supervision_sources else 0.0)
        for key, value in sorted(supervision_source_counter.items())
    }

    shape_total = segment_shape_counter.get("3_field", 0) + segment_shape_counter.get("4_field", 0)
    segment_shape_usage_ratio = {
        "3_field": segment_shape_counter.get("3_field", 0) / shape_total if shape_total else 0.0,
        "4_field": segment_shape_counter.get("4_field", 0) / shape_total if shape_total else 0.0,
    }

    return {
        "reciters": sorted(reciters),
        "supervision_source_usage_ratio": supervision_source_usage_ratio,
        "segment_shape_usage_ratio": segment_shape_usage_ratio,
        "segment_source_type_counts": dict(segment_source_type_counter),
    }


def _filter_dataset_by_support(
    dataset: dict[tuple[str, int, int], list[dict[str, Any]]],
    *,
    supported: bool,
) -> dict[tuple[str, int, int], list[dict[str, Any]]]:
    out: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for (reciter_id, surah, ayah), words in dataset.items():
        if is_qcom_word_supervision_supported(reciter_id) == supported:
            out[(reciter_id, surah, ayah)] = words
    return out


def _evaluate_on_datasets(
    *,
    predictions: dict[tuple[str, int, int], list[dict[str, Any]]],
    reference: dict[tuple[str, int, int], list[dict[str, Any]]],
) -> dict[str, Any]:
    boundary_errors_ms: list[float] = []
    per_reciter_errors: dict[str, list[float]] = defaultdict(list)
    missing_predictions: list[str] = []
    matched_ayahs = 0

    for key, reference_words in reference.items():
        reciter_id, surah, ayah = key
        pred_words = predictions.get(key)
        if pred_words is None:
            missing_predictions.append(f"{reciter_id}:{surah}:{ayah}")
            continue

        reference_lookup = _word_lookup(reference_words)
        pred_lookup = _word_lookup(pred_words)

        for word_idx, reference_word in reference_lookup.items():
            pred_word = pred_lookup.get(word_idx)
            if pred_word is None:
                continue
            try:
                reference_start = float(reference_word.get("start_s"))
                reference_end = float(reference_word.get("end_s"))
                pred_start = float(pred_word.get("start_s"))
                pred_end = float(pred_word.get("end_s"))
            except (TypeError, ValueError):
                continue

            start_err = abs(pred_start - reference_start) * 1000.0
            end_err = abs(pred_end - reference_end) * 1000.0
            boundary_errors_ms.extend([start_err, end_err])
            per_reciter_errors[reciter_id].extend([start_err, end_err])

        matched_ayahs += 1

    summary_metrics = _metrics_from_errors(boundary_errors_ms)
    per_reciter = {
        reciter_id: _metrics_from_errors(values).to_dict()
        for reciter_id, values in sorted(per_reciter_errors.items())
    }
    return {
        "reference_ayahs": len(reference),
        "pred_ayahs": len(predictions),
        "matched_ayahs": matched_ayahs,
        "missing_predictions": missing_predictions,
        "summary": summary_metrics.to_dict(),
        "per_reciter": per_reciter,
    }


def evaluate_bakeoff(
    *,
    pred_dir: str | Path,
    reference_dir: str | Path,
) -> dict[str, Any]:
    predictions = _collect_dataset(pred_dir)
    reference = _collect_dataset(reference_dir)

    overall = _evaluate_on_datasets(predictions=predictions, reference=reference)
    supported_predictions = _filter_dataset_by_support(predictions, supported=True)
    supported_reference = _filter_dataset_by_support(reference, supported=True)
    unsupported_predictions = _filter_dataset_by_support(predictions, supported=False)
    unsupported_reference = _filter_dataset_by_support(reference, supported=False)

    supported_eval = _evaluate_on_datasets(
        predictions=supported_predictions, reference=supported_reference
    )
    unsupported_eval = _evaluate_on_datasets(
        predictions=unsupported_predictions, reference=unsupported_reference
    )

    metadata = _collect_prediction_metadata(pred_dir)
    supported_reciters = sorted({reciter_id for (reciter_id, _, _) in supported_reference})
    unsupported_reciters = sorted({reciter_id for (reciter_id, _, _) in unsupported_reference})

    return {
        "overall": overall,
        "ayah_truth_everyayah": overall["summary"],
        "word_truth_qcom_segments_supported": supported_eval["summary"],
        "word_truth_model_only_unsupported": unsupported_eval["summary"],
        "supported_group": supported_eval,
        "unsupported_group": unsupported_eval,
        "coverage": {
            "supported_reciters": len(supported_reciters),
            "unsupported_reciters": len(unsupported_reciters),
            "supported_reciter_ids": supported_reciters,
            "unsupported_reciter_ids": unsupported_reciters,
        },
        "supervision_source_usage_ratio": metadata["supervision_source_usage_ratio"],
        "segment_shape_usage_ratio": metadata["segment_shape_usage_ratio"],
        "segment_source_type_counts": metadata["segment_source_type_counts"],
        "required_fields": {
            "median_p90_p95": {
                "median_abs_error_ms": overall["summary"]["median_abs_error_ms"],
                "p90_abs_error_ms": overall["summary"]["p90_abs_error_ms"],
                "p95_abs_error_ms": overall["summary"]["p95_abs_error_ms"],
            },
            "hit_rates": {
                "hit_rate_20ms": overall["summary"]["hit_rate_20ms"],
                "hit_rate_50ms": overall["summary"]["hit_rate_50ms"],
                "hit_rate_80ms": overall["summary"]["hit_rate_80ms"],
            },
            "per_reciter_metrics": overall["per_reciter"],
        },
    }
