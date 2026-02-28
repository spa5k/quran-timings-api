from __future__ import annotations

from pathlib import Path

import orjson

import quran_audio_data.benchmark_data as benchmark_data


def test_parse_everyayah_reciters() -> None:
    catalog = {
        "ayahCount": [1] * 114,
        "1": {"subfolder": "Abdul_Basit_Murattal_64kbps", "name": "Abdul Basit", "bitrate": "64kbps"},
        "2": {"subfolder": "Hudhaify_128kbps", "name": "Hudhaify", "bitrate": "128kbps"},
    }

    reciters = benchmark_data.parse_everyayah_reciters(catalog)
    assert len(reciters) == 2
    assert reciters[0].reciter_key == 1
    assert reciters[0].subfolder == "Abdul_Basit_Murattal_64kbps"


def test_prepare_benchmark_data_without_audio_download(tmp_path, monkeypatch) -> None:
    catalog = {
        "ayahCount": [7] + [1] * 113,
        "1": {
            "subfolder": "Abdul_Basit_Murattal_64kbps",
            "name": "Abdul Basit",
            "bitrate": "64kbps",
        },
    }
    verse_payload = {
        "verse": {
            "verse_key": "1:1",
            "words": [
                {"position": 1, "text_uthmani": "بِسْمِ"},
                {"position": 2, "text_uthmani": "ٱللَّهِ"},
            ],
        }
    }

    def fake_get(url, *args, **kwargs):
        if url == benchmark_data.EVERYAYAH_RECITATIONS_URL:
            return catalog
        if url.startswith("https://api.quran.com/api/v4/verses/by_key/"):
            return verse_payload
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(benchmark_data, "get_json_with_retry", fake_get)

    output = benchmark_data.prepare_benchmark_data(
        out_dir=tmp_path,
        count=1,
        ayah_keys=["1:1"],
        reciter_key=1,
        download_audio=False,
        seed=1,
    )

    manifest_path = Path(output["manifest_path"])
    assert manifest_path.exists()

    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "abdul_basit_murattal_64kbps" in lines[1]

    reference_files = sorted((tmp_path / "reference_templates").glob("*.json"))
    assert len(reference_files) == 1
    reference = orjson.loads(reference_files[0].read_bytes())
    assert reference["meta"]["verse_key"] == "1:1"
    assert len(reference["words"]) == 2


def test_fetch_quran_com_verse_retries_transient_timeout(monkeypatch) -> None:
    captured: dict[str, object] = {}
    verse_payload = {
        "verse": {
            "verse_key": "1:1",
            "words": [{"position": 1, "text_uthmani": "بِسْمِ"}],
        }
    }

    def fake_get(*, url: str, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return verse_payload

    monkeypatch.setattr(benchmark_data, "get_json_with_retry", fake_get)

    verse = benchmark_data.fetch_quran_com_verse(
        surah=1,
        ayah=1,
        timeout_s=0.01,
        retries=2,
        retry_backoff_s=0.0,
    )
    assert captured["retries"] == 2
    assert captured["retry_backoff_s"] == 0.0
    assert captured["timeout_s"] == 0.01
    assert str(captured["url"]).endswith("/1:1")
    assert verse["verse_key"] == "1:1"


def test_prepare_benchmark_data_resume_uses_existing_reference(tmp_path, monkeypatch) -> None:
    catalog = {
        "ayahCount": [7] + [1] * 113,
        "1": {
            "subfolder": "Abdul_Basit_Murattal_64kbps",
            "name": "Abdul Basit",
            "bitrate": "64kbps",
        },
    }

    reciter_id = "abdul_basit_murattal_64kbps"
    reference_dir = tmp_path / "reference_templates"
    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_path = reference_dir / f"{reciter_id}_s001_a001.json"
    reference_payload = {
        "meta": {
            "reciter_id": reciter_id,
            "surah": 1,
            "ayah": 1,
            "verse_key": "1:1",
        },
        "words": [
            {
                "ayah": 1,
                "word_index_in_ayah": 1,
                "text_uthmani": "بِسْمِ",
                "start_s": None,
                "end_s": None,
            }
        ],
    }
    reference_path.write_bytes(orjson.dumps(reference_payload))

    def fake_get(url, *args, **kwargs):
        if url == benchmark_data.EVERYAYAH_RECITATIONS_URL:
            return catalog
        if url.startswith("https://api.quran.com/api/v4/verses/by_key/"):
            raise AssertionError("Quran.com should not be called when resuming with existing reference")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(benchmark_data, "get_json_with_retry", fake_get)

    output = benchmark_data.prepare_benchmark_data(
        out_dir=tmp_path,
        count=1,
        ayah_keys=["1:1"],
        reciter_key=1,
        download_audio=False,
        resume=True,
    )

    manifest_path = Path(output["manifest_path"])
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "1,1" in lines[1]


def test_prepare_benchmark_data_uses_manifest_reciter_id_override(tmp_path, monkeypatch) -> None:
    catalog = {
        "ayahCount": [7] + [1] * 113,
        "1": {
            "subfolder": "Abdul_Basit_Murattal_64kbps",
            "name": "Abdul Basit",
            "bitrate": "64kbps",
        },
    }
    verse_payload = {
        "verse": {
            "verse_key": "1:1",
            "words": [
                {"position": 1, "text_uthmani": "بِسْمِ"},
            ],
        }
    }

    def fake_get(url, *args, **kwargs):
        if url == benchmark_data.EVERYAYAH_RECITATIONS_URL:
            return catalog
        if url.startswith("https://api.quran.com/api/v4/verses/by_key/"):
            return verse_payload
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(benchmark_data, "get_json_with_retry", fake_get)

    output = benchmark_data.prepare_benchmark_data(
        out_dir=tmp_path,
        count=1,
        ayah_keys=["1:1"],
        reciter_key=1,
        manifest_reciter_id="abdurrahmaan_as-sudays",
        download_audio=False,
        seed=1,
    )

    manifest_path = Path(output["manifest_path"])
    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert "abdurrahmaan_as-sudays" in lines[1]
