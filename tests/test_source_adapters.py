from __future__ import annotations

from quran_audio_data.sources.adapters import normalize_payload_with_adapters
from quran_audio_data.text.quran_text import CanonicalWord, normalize_arabic


def _canonical_words() -> list[CanonicalWord]:
    return [
        CanonicalWord(
            surah=1,
            ayah=1,
            word_index_global=1,
            word_index_in_ayah=1,
            text_uthmani="بِسْمِ",
            text_norm=normalize_arabic("بِسْمِ"),
        ),
        CanonicalWord(
            surah=1,
            ayah=1,
            word_index_global=2,
            word_index_in_ayah=2,
            text_uthmani="ٱللَّهِ",
            text_norm=normalize_arabic("ٱللَّهِ"),
        ),
    ]


def test_cache_adapter_accepts_schema_payload() -> None:
    payload = {
        "schema_version": "v1",
        "audio": {"path": "n/a", "duration_s": 1.0, "sample_rate": 16000, "channels": 1},
        "meta": {"reciter_id": "test", "surah": 1, "input_mode": "ayah_file"},
        "engine": {"name": "existing", "model": "cache", "device": "n/a", "fallback_used": False},
        "ayahs": [{"surah": 1, "ayah": 1, "start_s": 0.0, "end_s": 1.0, "source": "existing"}],
        "words": [
            {
                "surah": 1,
                "ayah": 1,
                "word_index_global": 1,
                "word_index_in_ayah": 1,
                "text_uthmani": "بِسْمِ",
                "text_norm": normalize_arabic("بِسْمِ"),
                "start_s": 0.0,
                "end_s": 0.5,
            },
            {
                "surah": 1,
                "ayah": 1,
                "word_index_global": 2,
                "word_index_in_ayah": 2,
                "text_uthmani": "ٱللَّهِ",
                "text_norm": normalize_arabic("ٱللَّهِ"),
                "start_s": 0.5,
                "end_s": 1.0,
            },
        ],
        "qc": {"coverage": 1.0, "monotonic": True, "duration_match": True, "warnings": []},
    }
    bundle = normalize_payload_with_adapters(
        payload=payload,
        canonical_words=_canonical_words(),
        source_name="cache:/tmp/example.json",
        source_default="existing",
    )
    assert bundle is not None
    assert len(bundle.ayahs) == 1
    assert len(bundle.words) == 2
    assert bundle.words[0].engine_candidate == "existing"


def test_quranaudio_adapter_normalizes_verse_list_shape() -> None:
    payload = {
        "verses": [
            {
                "verse_key": "1:1",
                "start": 0.0,
                "end": 1.0,
                "words": [
                    {"position": 1, "word": "بسم", "start": 0.0, "end": 0.4},
                    {"position": 2, "word": "الله", "start": 0.4, "end": 1.0},
                ],
            }
        ]
    }
    bundle = normalize_payload_with_adapters(
        payload=payload,
        canonical_words=_canonical_words(),
        source_name="quranaudio:https://example.test",
        source_default="existing",
    )
    assert bundle is not None
    assert len(bundle.ayahs) == 1
    assert [word.word_index_in_ayah for word in bundle.words] == [1, 2]
    assert bundle.words[0].alignment_origin == "native"


def test_qf_adapter_normalizes_timestamp_segments() -> None:
    payload = {
        "audio_file": {
            "timestamps": [
                {
                    "verse_key": "1:1",
                    "timestamp_from": 1000,
                    "timestamp_to": 2000,
                    "segments": [
                        [1, 1000.0, 1400.0],
                        [2, 1400.0, 2000.0],
                    ],
                }
            ]
        }
    }
    bundle = normalize_payload_with_adapters(
        payload=payload,
        canonical_words=_canonical_words(),
        source_name="qf:https://api.quran.foundation/...",
        source_default="existing",
    )
    assert bundle is not None
    assert bundle.ayahs[0].start_s == 0.0
    assert bundle.ayahs[0].end_s == 1.0
    assert bundle.words[0].start_s == 0.0
    assert bundle.words[1].end_s == 1.0


def test_qf_adapter_preserves_chapter_timeline_for_multiple_ayahs() -> None:
    canonical_words = [
        CanonicalWord(
            surah=1,
            ayah=1,
            word_index_global=1,
            word_index_in_ayah=1,
            text_uthmani="أ",
            text_norm=normalize_arabic("أ"),
        ),
        CanonicalWord(
            surah=1,
            ayah=2,
            word_index_global=2,
            word_index_in_ayah=1,
            text_uthmani="ب",
            text_norm=normalize_arabic("ب"),
        ),
    ]
    payload = {
        "audio_file": {
            "timestamps": [
                {
                    "verse_key": "1:1",
                    "timestamp_from": 1000,
                    "timestamp_to": 2000,
                    "segments": [[1, 1000.0, 1500.0]],
                },
                {
                    "verse_key": "1:2",
                    "timestamp_from": 2000,
                    "timestamp_to": 3000,
                    "segments": [[1, 2000.0, 2500.0]],
                },
            ]
        }
    }
    bundle = normalize_payload_with_adapters(
        payload=payload,
        canonical_words=canonical_words,
        source_name="qf:https://api.quran.foundation/...",
        source_default="existing",
    )
    assert bundle is not None
    ayah_starts = [ayah.start_s for ayah in bundle.ayahs]
    assert ayah_starts == [0.0, 1.0]
    word_starts = [word.start_s for word in bundle.words]
    assert word_starts == [0.0, 1.0]


def test_source_url_adapter_normalizes_compact_ayah_map() -> None:
    payload = {"1": {"start": 0.0, "end": 1.0}}
    bundle = normalize_payload_with_adapters(
        payload=payload,
        canonical_words=_canonical_words(),
        source_name="source_url:https://example.test/compact.json",
        source_default="existing",
    )
    assert bundle is not None
    assert len(bundle.ayahs) == 1
    assert len(bundle.words) == 2
    assert bundle.words[0].alignment_origin == "distributed"
