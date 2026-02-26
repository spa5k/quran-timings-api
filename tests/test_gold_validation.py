from __future__ import annotations

import orjson

from quran_audio_data.evaluation import validate_gold_annotations


def test_validate_gold_annotations_passes_for_fully_labeled_file(tmp_path) -> None:
    payload = {
        "meta": {"reciter_id": "test", "surah": 1, "ayah": 1},
        "words": [
            {
                "ayah": 1,
                "word_index_in_ayah": 1,
                "text_uthmani": "بِسْمِ",
                "start_s": 0.0,
                "end_s": 0.2,
            },
            {
                "ayah": 1,
                "word_index_in_ayah": 2,
                "text_uthmani": "ٱللَّهِ",
                "start_s": 0.2,
                "end_s": 0.45,
            },
        ],
    }
    gold_file = tmp_path / "gold_ok.json"
    gold_file.write_bytes(orjson.dumps(payload))

    report = validate_gold_annotations(gold_dir=tmp_path)
    assert report["passes"] is True
    assert report["valid_files"] == 1
    assert report["invalid_files"] == 0
    assert report["unlabeled_words"] == 0
    assert report["invalid_duration_words"] == 0
    assert report["non_monotonic_words"] == 0


def test_validate_gold_annotations_flags_incomplete_and_invalid_words(tmp_path) -> None:
    payload = {
        "meta": {"reciter_id": "test", "surah": 2, "ayah": 5},
        "words": [
            {
                "ayah": 5,
                "word_index_in_ayah": 1,
                "text_uthmani": "لَا",
                "start_s": None,
                "end_s": None,
            },
            {
                "ayah": 5,
                "word_index_in_ayah": 1,
                "text_uthmani": "رَيْبَ",
                "start_s": 0.25,
                "end_s": 0.2,
            },
            {
                "ayah": 5,
                "word_index_in_ayah": 3,
                "text_uthmani": "فِيهِ",
                "start_s": 0.2,
                "end_s": 0.35,
            },
        ],
    }
    gold_file = tmp_path / "gold_bad.json"
    gold_file.write_bytes(orjson.dumps(payload))

    report = validate_gold_annotations(gold_dir=tmp_path, max_errors=20)
    assert report["passes"] is False
    assert report["valid_files"] == 0
    assert report["invalid_files"] == 1
    assert report["unlabeled_words"] == 1
    assert report["duplicate_word_index"] == 1
    assert report["invalid_duration_words"] == 1
    assert report["non_monotonic_words"] == 1
    assert report["errors"]
