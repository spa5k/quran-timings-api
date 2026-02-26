from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
from statistics import median
from typing import Any

import orjson


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    ordered = sorted(values)
    rank = (len(ordered) - 1) * (p / 100.0)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    frac = rank - lower
    return ordered[lower] * (1.0 - frac) + ordered[upper] * frac


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


def _append_error(errors: list[str], message: str, max_errors: int) -> None:
    if len(errors) < max_errors:
        errors.append(message)


def _extract_key_and_words(payload: dict[str, Any], file_path: Path) -> list[tuple[tuple[str, int, int], list[dict[str, Any]]]]:
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


def _metrics_from_errors(boundary_errors_ms: list[float]) -> dict[str, float]:
    if not boundary_errors_ms:
        return {
            "median_abs_error_ms": 0.0,
            "p90_abs_error_ms": 0.0,
            "p95_abs_error_ms": 0.0,
            "hit_rate_20ms": 0.0,
            "hit_rate_50ms": 0.0,
            "hit_rate_80ms": 0.0,
        }

    total = len(boundary_errors_ms)
    return {
        "median_abs_error_ms": float(median(boundary_errors_ms)),
        "p90_abs_error_ms": float(_percentile(boundary_errors_ms, 90)),
        "p95_abs_error_ms": float(_percentile(boundary_errors_ms, 95)),
        "hit_rate_20ms": sum(1 for value in boundary_errors_ms if value <= 20.0) / total,
        "hit_rate_50ms": sum(1 for value in boundary_errors_ms if value <= 50.0) / total,
        "hit_rate_80ms": sum(1 for value in boundary_errors_ms if value <= 80.0) / total,
    }


def evaluate_predictions(
    *,
    pred_dir: str | Path,
    gold_dir: str | Path,
) -> dict[str, Any]:
    predictions = _collect_dataset(pred_dir)
    gold = _collect_dataset(gold_dir)

    boundary_errors_ms: list[float] = []
    per_reciter_errors: dict[str, list[float]] = defaultdict(list)
    missing_predictions: list[str] = []
    matched_ayahs = 0

    for key, gold_words in gold.items():
        reciter_id, surah, ayah = key
        pred_words = predictions.get(key)
        if pred_words is None:
            missing_predictions.append(f"{reciter_id}:{surah}:{ayah}")
            continue

        gold_lookup = _word_lookup(gold_words)
        pred_lookup = _word_lookup(pred_words)

        for word_idx, gold_word in gold_lookup.items():
            pred_word = pred_lookup.get(word_idx)
            if pred_word is None:
                continue
            try:
                gold_start = float(gold_word.get("start_s"))
                gold_end = float(gold_word.get("end_s"))
                pred_start = float(pred_word.get("start_s"))
                pred_end = float(pred_word.get("end_s"))
            except (TypeError, ValueError):
                continue

            start_err = abs(pred_start - gold_start) * 1000.0
            end_err = abs(pred_end - gold_end) * 1000.0
            boundary_errors_ms.extend([start_err, end_err])
            per_reciter_errors[reciter_id].extend([start_err, end_err])

        matched_ayahs += 1

    summary_metrics = _metrics_from_errors(boundary_errors_ms)
    per_reciter = {
        reciter_id: _metrics_from_errors(values)
        for reciter_id, values in sorted(per_reciter_errors.items())
    }

    report = {
        "gold_ayahs": len(gold),
        "pred_ayahs": len(predictions),
        "matched_ayahs": matched_ayahs,
        "missing_predictions": missing_predictions,
        "summary": summary_metrics,
        "per_reciter": per_reciter,
        "targets_ms": {
            "median": 50.0,
            "p95": 120.0,
            "per_reciter_median": 65.0,
        },
        "passes_targets": (
            summary_metrics["median_abs_error_ms"] <= 50.0
            and summary_metrics["p95_abs_error_ms"] <= 120.0
            and all(
                metrics["median_abs_error_ms"] <= 65.0
                for metrics in per_reciter.values()
            )
        ),
    }

    return report


def validate_gold_annotations(
    *,
    gold_dir: str | Path,
    max_errors: int = 50,
) -> dict[str, Any]:
    root = Path(gold_dir)
    files = sorted(root.rglob("*.json")) if root.is_dir() else [root]

    errors: list[str] = []
    total_files = len(files)
    valid_files = 0
    invalid_files = 0
    parse_errors = 0
    no_words_files = 0
    total_words = 0
    unlabeled_words = 0
    invalid_duration_words = 0
    non_monotonic_words = 0
    missing_word_index = 0
    duplicate_word_index = 0

    for file_path in files:
        try:
            payload = orjson.loads(file_path.read_bytes())
        except Exception as exc:
            invalid_files += 1
            parse_errors += 1
            _append_error(errors, f"{file_path}: parse error: {exc}", max_errors=max_errors)
            continue

        if not isinstance(payload, dict):
            invalid_files += 1
            parse_errors += 1
            _append_error(errors, f"{file_path}: root payload is not an object", max_errors=max_errors)
            continue

        words = payload.get("words")
        if not isinstance(words, list) or not words:
            invalid_files += 1
            no_words_files += 1
            _append_error(errors, f"{file_path}: missing or empty words list", max_errors=max_errors)
            continue

        file_invalid = False
        previous_start = -1.0
        previous_end = -1.0
        seen_word_indices: set[int] = set()

        for idx, word in enumerate(words, start=1):
            total_words += 1
            if not isinstance(word, dict):
                file_invalid = True
                _append_error(
                    errors,
                    f"{file_path}: word[{idx}] is not an object",
                    max_errors=max_errors,
                )
                continue

            word_index = _as_int(word.get("word_index_in_ayah"))
            if word_index is None:
                file_invalid = True
                missing_word_index += 1
                _append_error(
                    errors,
                    f"{file_path}: word[{idx}] missing integer word_index_in_ayah",
                    max_errors=max_errors,
                )
            elif word_index in seen_word_indices:
                file_invalid = True
                duplicate_word_index += 1
                _append_error(
                    errors,
                    f"{file_path}: duplicate word_index_in_ayah={word_index}",
                    max_errors=max_errors,
                )
            else:
                seen_word_indices.add(word_index)

            start_s = _as_float(word.get("start_s"))
            end_s = _as_float(word.get("end_s"))
            if start_s is None or end_s is None:
                file_invalid = True
                unlabeled_words += 1
                _append_error(
                    errors,
                    f"{file_path}: word[{idx}] has null/non-numeric start_s/end_s",
                    max_errors=max_errors,
                )
                continue

            if end_s <= start_s:
                file_invalid = True
                invalid_duration_words += 1
                _append_error(
                    errors,
                    f"{file_path}: word[{idx}] end_s ({end_s}) must be > start_s ({start_s})",
                    max_errors=max_errors,
                )

            if start_s < previous_start or end_s < previous_end:
                file_invalid = True
                non_monotonic_words += 1
                _append_error(
                    errors,
                    (
                        f"{file_path}: word[{idx}] non-monotonic timing "
                        f"(prev_start={previous_start}, prev_end={previous_end}, "
                        f"start={start_s}, end={end_s})"
                    ),
                    max_errors=max_errors,
                )

            previous_start = start_s
            previous_end = end_s

        if file_invalid:
            invalid_files += 1
        else:
            valid_files += 1

    report = {
        "gold_files": total_files,
        "valid_files": valid_files,
        "invalid_files": invalid_files,
        "parse_errors": parse_errors,
        "no_words_files": no_words_files,
        "total_words": total_words,
        "unlabeled_words": unlabeled_words,
        "invalid_duration_words": invalid_duration_words,
        "non_monotonic_words": non_monotonic_words,
        "missing_word_index": missing_word_index,
        "duplicate_word_index": duplicate_word_index,
        "max_errors": max_errors,
        "errors": errors,
        "passes": invalid_files == 0 and parse_errors == 0,
    }
    return report
