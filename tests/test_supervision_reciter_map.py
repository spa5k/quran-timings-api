import orjson

from quran_audio_data.supervision.reciter_map import is_reciter_enabled, resolve_reciter_mapping


def test_supported_reciter_mapping_has_qcom_id() -> None:
    mapping = resolve_reciter_mapping("abdurrahmaan_as-sudays")
    assert mapping.qcom_word_supervision_supported is True
    assert mapping.qcom_recitation_id is not None


def test_unsupported_reciter_branch_is_explicit() -> None:
    mapping = resolve_reciter_mapping("yasser_ad-dussary")
    assert mapping.qcom_word_supervision_supported is False
    assert mapping.qcom_recitation_id is None


def test_catalog_override_respected(tmp_path) -> None:
    catalog_path = tmp_path / "reciter_catalog.json"
    catalog_path.write_bytes(
        orjson.dumps(
            {
                "configured_reciters": [
                    {
                        "manifest_reciter_id": "foo_reciter",
                        "enabled": True,
                        "qcom_word_supervision_supported": True,
                        "everyayah": {"subfolder": "Foo_64kbps"},
                        "quran_com": {"recitation_id": 99},
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
