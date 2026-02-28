from __future__ import annotations

import orjson

from quran_audio_data.gold_labeling import auto_label_gold_from_quran_com
import quran_audio_data.gold_labeling as gold_labeling


def test_auto_label_gold_from_quran_com_fills_word_boundaries(tmp_path, monkeypatch) -> None:
    gold_file = tmp_path / "gold.json"
    payload = {
        "meta": {"surah": 1, "ayah": 1, "reciter_id": "abdul_basit"},
        "words": [
            {"word_index_in_ayah": 1, "text_uthmani": "بِسْمِ", "start_s": None, "end_s": None},
            {"word_index_in_ayah": 2, "text_uthmani": "ٱللَّهِ", "start_s": None, "end_s": None},
        ],
    }
    gold_file.write_bytes(orjson.dumps(payload))

    api_payload = {
        "audio_file": {
            "timestamps": [
                {
                    "verse_key": "1:1",
                    "segments": [[1, 10.0, 200.0], [2, 210.0, 440.0]],
                }
            ]
        }
    }

    def fake_get(*, url: str, **kwargs):
        assert "chapter_recitations/2/1" in url
        return api_payload

    monkeypatch.setattr(gold_labeling, "get_json_with_retry", fake_get)

    summary = auto_label_gold_from_quran_com(
        gold_dir=tmp_path,
        chapter_reciter_id=2,
        request_retries=0,
        retry_backoff_s=0.0,
    )

    assert summary.files_updated == 1
    assert summary.words_labeled == 2
    assert summary.passes is True

    updated = orjson.loads(gold_file.read_bytes())
    assert updated["words"][0]["start_s"] == 0.01
    assert updated["words"][0]["end_s"] == 0.2
    assert updated["words"][1]["start_s"] == 0.21
    assert updated["words"][1]["end_s"] == 0.44


def test_auto_label_gold_from_quran_com_skips_already_labeled(tmp_path, monkeypatch) -> None:
    gold_file = tmp_path / "gold.json"
    payload = {
        "meta": {"surah": 1, "ayah": 1, "reciter_id": "abdul_basit"},
        "words": [
            {"word_index_in_ayah": 1, "text_uthmani": "بِسْمِ", "start_s": 0.01, "end_s": 0.2},
            {"word_index_in_ayah": 2, "text_uthmani": "ٱللَّهِ", "start_s": 0.21, "end_s": 0.44},
        ],
    }
    gold_file.write_bytes(orjson.dumps(payload))

    called = {"count": 0}

    def fake_get(url, *args, **kwargs):
        called["count"] += 1
        raise AssertionError("Should not fetch segments for already-labeled file")

    monkeypatch.setattr(gold_labeling, "get_json_with_retry", fake_get)

    summary = auto_label_gold_from_quran_com(
        gold_dir=tmp_path,
        chapter_reciter_id=2,
        request_retries=0,
        retry_backoff_s=0.0,
    )

    assert summary.files_updated == 0
    assert summary.files_skipped_already_labeled == 1
    assert summary.passes is True
    assert called["count"] == 0


def test_auto_label_gold_from_quran_com_interpolates_unmapped_words(tmp_path, monkeypatch) -> None:
    gold_file = tmp_path / "gold.json"
    payload = {
        "meta": {"surah": 1, "ayah": 1, "reciter_id": "abdul_basit"},
        "words": [
            {"word_index_in_ayah": 1, "text_uthmani": "أ", "start_s": None, "end_s": None},
            {"word_index_in_ayah": 2, "text_uthmani": "ب", "start_s": None, "end_s": None},
            {"word_index_in_ayah": 3, "text_uthmani": "ج", "start_s": None, "end_s": None},
        ],
    }
    gold_file.write_bytes(orjson.dumps(payload))

    # Position 2 intentionally missing in API segments.
    api_payload = {
        "audio_file": {
            "timestamps": [
                {
                    "verse_key": "1:1",
                    "timestamp_from": 0,
                    "timestamp_to": 300,
                    "segments": [[1, 0.0, 90.0], [3, 190.0, 300.0]],
                }
            ]
        }
    }

    def fake_get(*, url: str, **kwargs):
        return api_payload

    monkeypatch.setattr(gold_labeling, "get_json_with_retry", fake_get)

    summary = auto_label_gold_from_quran_com(
        gold_dir=tmp_path,
        chapter_reciter_id=2,
        request_retries=0,
        retry_backoff_s=0.0,
    )

    assert summary.passes is True
    assert summary.words_labeled == 3
    assert summary.words_missing_segments == 0

    updated = orjson.loads(gold_file.read_bytes())
    second = updated["words"][1]
    assert second["start_s"] is not None
    assert second["end_s"] is not None
    assert second["end_s"] > second["start_s"]


def test_auto_label_gold_from_quran_com_normalizes_chapter_absolute_ms(tmp_path, monkeypatch) -> None:
    gold_file = tmp_path / "gold.json"
    payload = {
        "meta": {"surah": 1, "ayah": 2, "reciter_id": "abdul_basit"},
        "words": [
            {"word_index_in_ayah": 1, "text_uthmani": "ٱلْحَمْدُ", "start_s": None, "end_s": None},
        ],
    }
    gold_file.write_bytes(orjson.dumps(payload))

    # Chapter-absolute segment: 7000->7300 with ayah anchor at 6300.
    # Ayah-relative should be ~0.7->1.0.
    api_payload = {
        "audio_file": {
            "timestamps": [
                {
                    "verse_key": "1:2",
                    "timestamp_from": 6300,
                    "timestamp_to": 16440,
                    "segments": [[1, 7000.0, 7300.0]],
                }
            ]
        }
    }

    def fake_get(*, url: str, **kwargs):
        return api_payload

    monkeypatch.setattr(gold_labeling, "get_json_with_retry", fake_get)

    summary = auto_label_gold_from_quran_com(
        gold_dir=tmp_path,
        chapter_reciter_id=2,
        request_retries=0,
        retry_backoff_s=0.0,
    )

    assert summary.passes is True
    updated = orjson.loads(gold_file.read_bytes())
    assert updated["words"][0]["start_s"] == 0.7
    assert updated["words"][0]["end_s"] == 1.0
