from __future__ import annotations

import json
from pathlib import Path

from quran_audio_data.ui_sync import RunCandidate, update_catalog


def test_update_catalog_ignores_non_numeric_surah_values(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(
            {
                "recitations": [
                    {
                        "id": "reciter-1",
                        "surahs": [
                            {"surah": "x", "title": "Legacy bad row"},
                        ],
                    }
                ]
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    artifact = tmp_path / "reciter-1_s001_full.json"
    artifact.write_text("{}", encoding="utf-8")
    latest = {
        ("reciter-1", 1): RunCandidate(
            reciter_id="reciter-1",
            surah=1,
            path=artifact,
            mtime_ns=artifact.stat().st_mtime_ns,
        )
    }

    changed, added, updated, skipped = update_catalog(
        catalog_path=catalog_path,
        latest_candidates=latest,
        dry_run=False,
    )

    assert changed
    assert added == 1
    assert updated == 0
    assert skipped == 0

    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    surahs = payload["recitations"][0]["surahs"]
    assert any(entry.get("surah") == 1 for entry in surahs if isinstance(entry, dict))
