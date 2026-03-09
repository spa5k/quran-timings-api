import orjson

from quran_audio_data.supervision.reciter_map import is_reciter_enabled, resolve_reciter_mapping


def test_supported_reciter_mapping_has_qcom_id() -> None:
    mapping = resolve_reciter_mapping("abdurrahmaan_as-sudays")
    assert mapping.qcom_word_supervision_supported is True
    assert mapping.qcom_recitation_id is not None
    assert mapping.everyayah_subfolder == "Abdurrahmaan_As-Sudais_192kbps"


def test_unsupported_reciter_branch_is_explicit() -> None:
    mapping = resolve_reciter_mapping("yasser_ad-dussary")
    assert mapping.qcom_word_supervision_supported is False
    assert mapping.qcom_recitation_id is None
    assert mapping.everyayah_subfolder == "Yasser_Ad-Dussary_128kbps"


def test_canonical_everyayah_reciter_mapping_is_source_scoped() -> None:
    mapping = resolve_reciter_mapping("eya_abdurrahmaan_as_sudais_192kbps")
    assert mapping.everyayah_subfolder == "Abdurrahmaan_As-Sudais_192kbps"
    assert mapping.qcom_word_supervision_supported is False
    assert mapping.qcom_recitation_id is None


def test_catalog_override_respected(tmp_path) -> None:
    catalog_path = tmp_path / "reciter_catalog.json"
    catalog_path.write_bytes(
        orjson.dumps(
            {
                "reciters": [
                    {
                        "slug": "foo_reciter",
                        "enabled": True,
                        "capabilities": {
                            "ayah_by_ayah": True,
                            "word_by_word": True,
                        },
                        "source": {
                            "everyayah": {"subfolder": "Foo_64kbps"},
                            "quran_com": {"recitation_id": 99},
                        },
                    }
                ]
            }
        )
    )

    mapping = resolve_reciter_mapping("foo_reciter", catalog_path=catalog_path)
    assert mapping.everyayah_subfolder == "Foo_64kbps"
    assert mapping.qcom_recitation_id == 99
    assert mapping.qcom_word_supervision_supported is True
    assert is_reciter_enabled("foo_reciter", catalog_path=catalog_path) is True
