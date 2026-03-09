from __future__ import annotations

import quran_audio_data.reciters as reciters_module

from quran_audio_data.reciters import (
    fetch_quranicaudio_reciters,
    get_reciter,
    list_reciters,
    normalize_reciter_id,
    prefill_registry_from_sources,
    reciter_exists,
    source_scoped_reciter_id,
    upsert_reciter,
)


def test_upsert_reciter_normalizes_and_updates(tmp_path) -> None:
    reciters_path = tmp_path / "reciters.json"

    created = upsert_reciter(
        reciter_id="Abdullaah 3Awwaad Al-Juhaynee",
        name="Abdullaah 3Awwaad Al-Juhaynee",
        source="quranicaudio",
        notes="first",
        path=reciters_path,
    )
    assert created["id"] == "abdullaah_3awwaad_al_juhaynee"
    assert created["name"] == "Abdullaah 3Awwaad Al-Juhaynee (quranicaudio)"

    updated = upsert_reciter(
        reciter_id="abdullaah_3awwaad_al_juhaynee",
        name="Abdullaah 3Awwaad Al-Juhaynee",
        source="custom",
        notes="second",
        path=reciters_path,
    )

    rows = list_reciters(reciters_path)
    assert len(rows) == 1
    assert updated["id"] == "abdullaah_3awwaad_al_juhaynee"
    assert updated["source"] == "custom"
    assert updated["name"] == "Abdullaah 3Awwaad Al-Juhaynee (custom)"
    assert get_reciter("Abdullaah 3Awwaad Al-Juhaynee", reciters_path) is not None


def test_reciter_exists(tmp_path) -> None:
    reciters_path = tmp_path / "reciters.json"
    upsert_reciter(
        reciter_id="legacy_reciter",
        name="Legacy Reciter",
        source="custom",
        path=reciters_path,
    )
    assert reciter_exists(
        "legacy_reciter",
        path=reciters_path,
    )
    assert not reciter_exists(
        "unknown_reciter",
        path=reciters_path,
    )


def test_normalize_reciter_id() -> None:
    assert (
        normalize_reciter_id(" Abdullaah/3awwaad-Al Juhaynee ") == "abdullaah_3awwaad_al_juhaynee"
    )


def test_source_scoped_reciter_id() -> None:
    assert (
        source_scoped_reciter_id("quranicaudio", "abdullaah_3awwaad_al-juhaynee")
        == "quranicaudio_abdullaah_3awwaad_al_juhaynee"
    )
    assert source_scoped_reciter_id("quran.com", 7) == "quran_com_7"


def test_fetch_quranicaudio_reciters_parses_homepage_payload(monkeypatch) -> None:
    homepage = (
        '...qaris:[{id:1,name:"Abdullah Awad al-Juhani",arabic_name:"x",'
        'relative_path:"abdullaah_3awwaad_al-juhaynee/",file_formats:"mp3"},'
        '{id:2,name:"Sa`ud ash-Shuraym",arabic_name:"y",'
        'relative_path:"sa3ood_al-shuraym/",file_formats:"mp3"}],foo:bar...'
    )
    monkeypatch.setattr(
        reciters_module,
        "get_bytes_with_retry",
        lambda **kwargs: homepage.encode("utf-8"),
    )
    rows = fetch_quranicaudio_reciters()
    assert len(rows) == 2
    assert rows[0]["id"] == "quranicaudio_abdullaah_3awwaad_al_juhaynee"
    assert rows[0]["source"] == "quranicaudio"


def test_prefill_registry_from_sources_writes_merged_rows(tmp_path, monkeypatch) -> None:
    reciters_path = tmp_path / "reciters.json"
    monkeypatch.setattr(
        reciters_module,
        "fetch_quranicaudio_reciters",
        lambda: [
            {
                "id": "quranicaudio_abdurrahmaan_as_sudays",
                "name": "Abdurrahmaan As-Sudays",
                "source": "quranicaudio",
                "notes": "qa",
            }
        ],
    )
    monkeypatch.setattr(
        reciters_module,
        "fetch_everyayah_reciters",
        lambda: [
            {
                "id": "everyayah_abdurrahmaan_as_sudays",
                "name": "As Sudays",
                "source": "everyayah",
                "notes": "ea",
            }
        ],
    )
    monkeypatch.setattr(
        reciters_module,
        "fetch_quran_com_reciters",
        lambda: [
            {
                "id": "quran_com_1",
                "name": "Quran.com 1",
                "source": "quran.com",
                "notes": "qc",
            }
        ],
    )

    payload = prefill_registry_from_sources(reciters_path, preserve_existing=False)
    ids = [str(item.get("id")) for item in payload.get("reciters", [])]
    assert "quranicaudio_abdurrahmaan_as_sudays" in ids
    assert "everyayah_abdurrahmaan_as_sudays" in ids
    assert "quran_com_1" in ids
