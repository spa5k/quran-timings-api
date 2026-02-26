from __future__ import annotations

from pathlib import Path

import orjson
import requests

import quran_audio_data.benchmark_data as benchmark_data


class _FakeResponse:
    def __init__(self, payload=None, content: bytes = b"", status_code: int = 200):
        self._payload = payload
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


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
            return _FakeResponse(payload=catalog)
        if url.startswith("https://api.quran.com/api/v4/verses/by_key/"):
            return _FakeResponse(payload=verse_payload)
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(benchmark_data.requests, "get", fake_get)

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

    gold_files = sorted((tmp_path / "gold_templates").glob("*.json"))
    assert len(gold_files) == 1
    gold = orjson.loads(gold_files[0].read_bytes())
    assert gold["meta"]["verse_key"] == "1:1"
    assert len(gold["words"]) == 2


def test_fetch_quran_com_verse_retries_transient_timeout(monkeypatch) -> None:
    calls = {"count": 0}
    verse_payload = {
        "verse": {
            "verse_key": "1:1",
            "words": [{"position": 1, "text_uthmani": "بِسْمِ"}],
        }
    }

    def fake_get(url, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] < 3:
            raise requests.exceptions.ReadTimeout(f"timeout {url}")
        return _FakeResponse(payload=verse_payload)

    monkeypatch.setattr(benchmark_data.requests, "get", fake_get)
    monkeypatch.setattr(benchmark_data.time, "sleep", lambda *_args, **_kwargs: None)

    verse = benchmark_data.fetch_quran_com_verse(
        surah=1,
        ayah=1,
        timeout_s=0.01,
        retries=2,
        retry_backoff_s=0.0,
    )
    assert calls["count"] == 3
    assert verse["verse_key"] == "1:1"


def test_prepare_benchmark_data_resume_uses_existing_gold(tmp_path, monkeypatch) -> None:
    catalog = {
        "ayahCount": [7] + [1] * 113,
        "1": {
            "subfolder": "Abdul_Basit_Murattal_64kbps",
            "name": "Abdul Basit",
            "bitrate": "64kbps",
        },
    }

    reciter_id = "abdul_basit_murattal_64kbps"
    gold_dir = tmp_path / "gold_templates"
    gold_dir.mkdir(parents=True, exist_ok=True)
    gold_path = gold_dir / f"{reciter_id}_s001_a001.json"
    gold_payload = {
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
    gold_path.write_bytes(orjson.dumps(gold_payload))

    def fake_get(url, *args, **kwargs):
        if url == benchmark_data.EVERYAYAH_RECITATIONS_URL:
            return _FakeResponse(payload=catalog)
        if url.startswith("https://api.quran.com/api/v4/verses/by_key/"):
            raise AssertionError("Quran.com should not be called when resuming with existing gold")
        raise AssertionError(f"Unexpected URL: {url}")

    monkeypatch.setattr(benchmark_data.requests, "get", fake_get)

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
