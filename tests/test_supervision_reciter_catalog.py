from __future__ import annotations

from pathlib import Path

import orjson

import quran_audio_data.supervision.reciter_catalog as reciter_catalog


def test_write_reciter_catalog_merges_sources(tmp_path, monkeypatch) -> None:
    everyayah_payload = {
        "ayahCount": [7] + [1] * 113,
        "1": {
            "name": "Abdul Basit Murattal",
            "subfolder": "Abdul_Basit_Murattal_64kbps",
            "bitrate": "64kbps",
        },
    }
    qcom_payload = {
        "recitations": [
            {
                "id": 1,
                "reciter_name": "AbdulBaset AbdulSamad",
                "translated_name": {"name": "AbdulBaset AbdulSamad"},
                "style": {"name": "Murattal"},
            }
        ]
    }

    monkeypatch.setattr(reciter_catalog, "fetch_everyayah_catalog", lambda: everyayah_payload)
    monkeypatch.setattr(
        reciter_catalog, "fetch_recitation_catalog", lambda language="en": qcom_payload
    )

    output_path = tmp_path / "reciter_catalog.json"
    payload = reciter_catalog.write_reciter_catalog(
        path=output_path,
        enabled_reciters={"abdul_basit_murattal_64kbps"},
    )

    assert output_path.exists()
    loaded = orjson.loads(output_path.read_bytes())
    assert payload["counts"]["everyayah_source_reciters"] == 1
    assert payload["counts"]["quran_com_source_reciters"] == 1
    assert loaded["counts"]["enabled_reciters"] >= 1
    assert isinstance(loaded["reciters"], list)

    entry = reciter_catalog.get_configured_reciter_entry(
        "abdul_basit_murattal_64kbps",
        catalog_path=output_path,
    )
    assert entry is not None
    assert entry["enabled"] is True
    assert entry["check_type"] in {"both", "ayah_by_ayah", "word_by_word", "model_only"}
    assert str(entry.get("slug")) == "abdul_basit_murattal_64kbps"


def test_read_reciter_catalog_invalid_returns_none(tmp_path) -> None:
    target = Path(tmp_path / "invalid.json")
    target.write_text("{invalid json", encoding="utf-8")
    assert reciter_catalog.read_reciter_catalog(target) is None
