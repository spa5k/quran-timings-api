from __future__ import annotations

from pathlib import Path

import orjson

from quran_audio_data.ui_sync import export_api_from_latest_runs


def _write_full_payload(path: Path, *, audio_path: Path) -> None:
    payload = {
        "schema_version": "v3",
        "audio": {
            "path": str(audio_path),
            "duration_s": 3.5,
            "sample_rate": 16000,
            "channels": 1,
        },
        "meta": {
            "reciter_id": "reciter-1",
            "surah": 1,
            "input_mode": "full_surah",
        },
        "engine": {"name": "nemo", "model": "fake", "device": "cpu", "fallback_used": False},
        "ayahs": [{"surah": 1, "ayah": 1, "start_s": 0.0, "end_s": 3.5, "source": "aligned"}],
        "words": [
            {
                "surah": 1,
                "ayah": 1,
                "word_index_global": 1,
                "word_index_in_ayah": 1,
                "text_uthmani": "الْحَمْدُ",
                "text_norm": "الحمد",
                "start_s": 0.0,
                "end_s": 1.0,
                "confidence": 0.98,
                "alignment_origin": "native",
                "match_score": 99.0,
                "engine_candidate": "nemo",
                "source_start_s": 0.01,
                "source_end_s": 0.99,
                "source_provider": "quran_com",
            },
            {
                "surah": 1,
                "ayah": 1,
                "word_index_global": 2,
                "word_index_in_ayah": 2,
                "text_uthmani": "لِلَّهِ",
                "text_norm": "لله",
                "start_s": 1.0,
                "end_s": 1.6,
                "confidence": 0.97,
                "alignment_origin": "native",
                "match_score": 98.0,
                "engine_candidate": "nemo",
            },
        ],
        "qc": {
            "coverage": 1.0,
            "monotonic": True,
            "duration_match": True,
            "warnings": ["boundary_refinement_applied", "actionable_warning"],
            "reason_codes": [],
            "speech_end_delta_ratio": 0.99,
            "interpolated_ratio": 0.0,
            "boundary_refine_method": "numpy",
        },
        "supervision_sources": [
            "everyayah:subfolder=Reciter_One_128kbps:surah=1:scope=full_surah",
            "qcom:qcom_verse:3:shape=4_field",
        ],
        "segment_source_type": "qcom_verse",
        "selected_candidate_engine": "nemo",
        "candidate_scores": {"nemo": 1.0},
        "pass_trace": ["strict_multi_pass"],
        "everyayah_stitch_eval": None,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def test_export_api_from_latest_runs_writes_split_contract(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    api_root = tmp_path / "data" / "api"
    reciters_index_path = tmp_path / "data" / "reciters.json"
    ui_data_dir = tmp_path / "ui" / "public" / "data"
    dist_data_dir = tmp_path / "ui" / "dist" / "data"

    audio_path = tmp_path / "audio" / "reciter-1_001.mp3"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake-mp3")

    full_json = runs_root / "reciter-1_s001" / "outputs" / "reciter-1_s001_full.json"
    _write_full_payload(full_json, audio_path=audio_path)

    reciters_index_path.parent.mkdir(parents=True, exist_ok=True)
    reciters_index_path.write_bytes(
        orjson.dumps(
            {
                "schema_version": "v1",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "counts": {
                    "everyayah_source_reciters": 0,
                    "quran_com_source_reciters": 0,
                    "configured_reciters": 1,
                    "enabled_reciters": 1,
                },
                "sources": {},
                "reciters": [
                    {
                        "slug": "reciter-1",
                        "name": "Reciter One",
                        "enabled": True,
                        "check_type": "word_by_word",
                        "capabilities": {"ayah_by_ayah": False, "word_by_word": True},
                        "source": {
                            "everyayah": {"subfolder": None, "reciter_key": None, "name": None},
                            "quran_com": {"recitation_id": 3, "name": "Reciter One"},
                        },
                        "endpoints": {"metadata": "/data/reciters/reciter-1/metadata.json"},
                        "surahs_available": [],
                        "surah_count": 0,
                    }
                ],
            },
            option=orjson.OPT_INDENT_2,
        )
    )

    summary = export_api_from_latest_runs(
        runs_root=runs_root,
        api_root=api_root,
        reciters_index_path=reciters_index_path,
        ui_data_dir=ui_data_dir,
        dist_data_dir=dist_data_dir,
        sync_dist=False,
        prune_ui=True,
        dry_run=False,
    )

    assert summary["keys_selected"] == 1
    assert (api_root / "reciters" / "reciter-1" / "surahs" / "1" / "metadata.json").exists()
    assert (api_root / "reciters" / "reciter-1" / "surahs" / "1" / "timings.json").exists()
    assert (api_root / "reciters" / "reciter-1" / "metadata.json").exists()
    assert not (api_root / "audio").exists()

    surah_metadata = orjson.loads(
        (api_root / "reciters" / "reciter-1" / "surahs" / "1" / "metadata.json").read_bytes()
    )
    assert surah_metadata["schema_version"] == "v2"
    assert surah_metadata["audio"]["granularity"] == "mixed"
    assert surah_metadata["audio"]["primary_asset"] == "qcom_surah"
    assert (
        surah_metadata["audio"]["assets"]["qcom_surah"]["url"]
        == "https://api.quran.com/api/v4/chapter_recitations/3/1"
    )
    assert (
        surah_metadata["audio"]["assets"]["everyayah_ayah"]["template"]
        == "https://everyayah.com/data/Reciter_One_128kbps/001{ayah:03d}.mp3"
    )
    assert surah_metadata["quality"]["warnings"] == ["actionable_warning"]
    assert surah_metadata["quality"]["boundary_refine_method"] == "numpy"
    assert "everyayah_stitch_eval" not in surah_metadata["quality"]

    surah_timings = orjson.loads(
        (api_root / "reciters" / "reciter-1" / "surahs" / "1" / "timings.json").read_bytes()
    )
    assert surah_timings["schema_version"] == "v2"
    assert "source" not in surah_timings["ayahs"][0]
    assert surah_timings["ayahs"][0]["audio_asset"] == "everyayah_ayah"
    assert surah_timings["ayahs"][0]["audio_key"] == "001"
    assert surah_timings["ayahs"][0]["audio_url"] == (
        "https://everyayah.com/data/Reciter_One_128kbps/001001.mp3"
    )
    assert set(surah_timings["words"][0]) == {
        "surah",
        "ayah",
        "word_index_global",
        "word_index_in_ayah",
        "text_uthmani",
        "text_norm",
        "start_s",
        "end_s",
        "source_start_s",
        "source_end_s",
    }
    assert surah_timings["words"][0]["source_start_s"] == 0.01
    assert surah_timings["words"][0]["source_end_s"] == 0.99
    assert set(surah_timings["words"][1]) == {
        "surah",
        "ayah",
        "word_index_global",
        "word_index_in_ayah",
        "text_uthmani",
        "text_norm",
        "start_s",
        "end_s",
    }

    reciters_payload = orjson.loads(reciters_index_path.read_bytes())
    reciter = reciters_payload["reciters"][0]
    assert reciter["surahs_available"] == [1]
    assert reciter["surah_count"] == 1

    served_reciters = ui_data_dir / "reciters.json"
    assert served_reciters.exists()
    assert (ui_data_dir / "reciters" / "reciter-1" / "surahs" / "1" / "timings.json").exists()
    assert not (ui_data_dir / "audio").exists()


def test_export_api_from_latest_runs_filters_selected_reciters_and_surahs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    api_root = tmp_path / "data" / "api"
    reciters_index_path = tmp_path / "data" / "reciters.json"
    ui_data_dir = tmp_path / "ui" / "public" / "data"
    dist_data_dir = tmp_path / "ui" / "dist" / "data"

    audio_path = tmp_path / "audio" / "sample.mp3"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake-mp3")

    _write_full_payload(
        runs_root / "reciter-1_s001" / "outputs" / "reciter-1_s001_full.json",
        audio_path=audio_path,
    )
    _write_full_payload(
        runs_root / "reciter-2_s002" / "outputs" / "reciter-2_s002_full.json",
        audio_path=audio_path,
    )

    reciters_index_path.parent.mkdir(parents=True, exist_ok=True)
    reciters_index_path.write_bytes(
        orjson.dumps(
            {
                "schema_version": "v1",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "counts": {"configured_reciters": 2, "enabled_reciters": 2},
                "sources": {},
                "reciters": [
                    {"slug": "reciter-1", "name": "Reciter One", "enabled": True},
                    {"slug": "reciter-2", "name": "Reciter Two", "enabled": True},
                ],
            },
            option=orjson.OPT_INDENT_2,
        )
    )

    summary = export_api_from_latest_runs(
        runs_root=runs_root,
        api_root=api_root,
        reciters_index_path=reciters_index_path,
        ui_data_dir=ui_data_dir,
        dist_data_dir=dist_data_dir,
        sync_dist=False,
        prune_ui=True,
        dry_run=False,
        include_reciters={"reciter-1"},
        include_surahs={1},
    )

    assert summary["keys_selected"] == 1
    assert (api_root / "reciters" / "reciter-1" / "surahs" / "1" / "metadata.json").exists()
    assert not (api_root / "reciters" / "reciter-2").exists()


def test_export_api_from_latest_runs_skips_unindexed_reciters(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    api_root = tmp_path / "data" / "api"
    reciters_index_path = tmp_path / "data" / "reciters.json"
    ui_data_dir = tmp_path / "ui" / "public" / "data"
    dist_data_dir = tmp_path / "ui" / "dist" / "data"

    audio_path = tmp_path / "audio" / "sample.mp3"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"fake-mp3")

    _write_full_payload(
        runs_root / "reciter-1_s001" / "outputs" / "reciter-1_s001_full.json",
        audio_path=audio_path,
    )
    _write_full_payload(
        runs_root
        / "random_smoke_reciter_20260310_s001"
        / "outputs"
        / "random_smoke_reciter_20260310_s001_full.json",
        audio_path=audio_path,
    )

    reciters_index_path.parent.mkdir(parents=True, exist_ok=True)
    reciters_index_path.write_bytes(
        orjson.dumps(
            {
                "schema_version": "v1",
                "generated_at": "2026-01-01T00:00:00+00:00",
                "counts": {"configured_reciters": 1, "enabled_reciters": 1},
                "sources": {},
                "reciters": [
                    {"slug": "reciter-1", "name": "Reciter One", "enabled": True},
                ],
            },
            option=orjson.OPT_INDENT_2,
        )
    )

    summary = export_api_from_latest_runs(
        runs_root=runs_root,
        api_root=api_root,
        reciters_index_path=reciters_index_path,
        ui_data_dir=ui_data_dir,
        dist_data_dir=dist_data_dir,
        sync_dist=False,
        prune_ui=True,
        dry_run=False,
    )

    assert summary["keys_selected"] == 1
    assert summary["keys_skipped_unindexed"] == 1
    assert (api_root / "reciters" / "reciter-1" / "surahs" / "1" / "metadata.json").exists()
    assert not (api_root / "reciters" / "random_smoke_reciter_20260310").exists()
