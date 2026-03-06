from __future__ import annotations

import json

import numpy as np
import orjson

from quran_audio_data.detect import (
    decode_ctc_greedy,
    infer_reciter_name_from_audio_url,
    load_quran_references,
    parse_reference_hint_from_audio_url,
    rank_ayah_candidates,
    rank_ayah_candidates_from_references,
    slugify_reciter_name,
    VerseReference,
)
from quran_audio_data.text.quran_text import QuranTextStore, normalize_arabic


def test_parse_reference_hint_from_audio_url_full_ayah_file() -> None:
    hint = parse_reference_hint_from_audio_url(
        "https://download.quranicaudio.com/quran/foo/001001.mp3"
    )
    assert hint.scope == "ayah_file"
    assert hint.surah == 1
    assert hint.ayah == 1


def test_parse_reference_hint_from_audio_url_surah_file() -> None:
    hint = parse_reference_hint_from_audio_url(
        "https://download.quranicaudio.com/quran/foo/001.mp3"
    )
    assert hint.scope == "surah_file"
    assert hint.surah == 1
    assert hint.ayah is None


def test_infer_reciter_name_from_audio_url() -> None:
    name = infer_reciter_name_from_audio_url(
        "https://download.quranicaudio.com/quran/abdullaah_3awwaad_al-juhaynee/001.mp3"
    )
    assert name == "abdullaah 3awwaad al juhaynee"


def test_slugify_reciter_name_consistent() -> None:
    assert slugify_reciter_name("Abdullaah 3Awwaad Al-Juhaynee") == "abdullaah_3awwaad_al_juhaynee"
    assert slugify_reciter_name("  ___ ") == "reciter"


def test_rank_ayah_candidates_picks_best_match(tmp_path) -> None:
    text_path = tmp_path / "quran.json"
    payload = {
        "surahs": {
            "1": {
                "1": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
                "2": "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
            }
        }
    }
    text_path.write_bytes(orjson.dumps(payload))
    store = QuranTextStore(text_path)

    candidates = rank_ayah_candidates(
        transcript_norm=normalize_arabic("بسم الله الرحمن الرحيم"),
        text_store=store,
        top_k=2,
    )
    assert len(candidates) == 2
    assert candidates[0].surah == 1
    assert candidates[0].ayah == 1
    assert candidates[0].score >= candidates[1].score


def test_rank_ayah_candidates_supports_multi_ayah_span() -> None:
    references = [
        VerseReference(surah=1, ayah=1, text_norm=normalize_arabic("بسم الله الرحمن الرحيم")),
        VerseReference(surah=1, ayah=2, text_norm=normalize_arabic("الحمد لله رب العالمين")),
        VerseReference(surah=1, ayah=3, text_norm=normalize_arabic("الرحمن الرحيم")),
        VerseReference(surah=2, ayah=1, text_norm=normalize_arabic("الم")),
    ]
    transcript_norm = normalize_arabic("بسم الله الرحمن الرحيم الحمد لله رب العالمين")
    candidates = rank_ayah_candidates_from_references(
        transcript_norm=transcript_norm,
        references=references,
        top_k=3,
        max_span=4,
    )
    assert candidates
    assert candidates[0].surah == 1
    assert candidates[0].ayah == 1
    assert candidates[0].ayah_end == 2


def test_decode_ctc_greedy_collapses_repeats_and_blank() -> None:
    # Time steps pick ids: 3, 3, 4, 0(blank), 4 => decoded should be "ابب"
    # (blank resets repetition in standard CTC greedy decoding)
    logits = np.array(
        [
            [0.0, 0.0, 0.0, 9.0, 1.0],
            [0.0, 0.0, 0.0, 9.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 9.0],
            [9.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 9.0],
        ],
        dtype=np.float32,
    )
    id_to_token = {0: "<blank>", 3: "ا", 4: "ب"}
    assert decode_ctc_greedy(logits=logits, id_to_token=id_to_token) == "ابب"


def test_load_quran_references_supports_keyed_payload(tmp_path) -> None:
    quran_path = tmp_path / "quran.json"
    quran_path.write_text(
        json.dumps(
            {
                "1:1": "بسم الله الرحمن الرحيم",
                "1:2": "الحمد لله رب العالمين",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # fallback store is still required by function signature, but should not be used here.
    text_path = tmp_path / "fallback_text.json"
    text_path.write_bytes(
        orjson.dumps({"surahs": {"1": {"1": "x"}}})
    )
    store = QuranTextStore(text_path)

    refs = load_quran_references(text_store=store, quran_path=quran_path)
    assert len(refs) == 2
    assert refs[0].surah == 1
    assert refs[0].ayah == 1
