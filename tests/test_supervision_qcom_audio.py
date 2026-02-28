from quran_audio_data.supervision.qcom_audio import (
    extract_chapter_timestamp_segments,
    extract_verse_segments,
    resolve_verse_audio_url,
)


def test_resolve_verse_audio_url_resolves_relative_path() -> None:
    resolved = resolve_verse_audio_url("audio/recitations/1/001001.mp3")
    assert resolved == "https://verses.quran.com/audio/recitations/1/001001.mp3"


def test_extract_chapter_timestamp_segments_returns_payload() -> None:
    payload = {
        "audio_file": {
            "timestamps": [
                {
                    "verse_key": "1:1",
                    "segments": [
                        [1, 0.0, 100.0],
                        [2, 110.0, 200.0],
                    ],
                }
            ]
        }
    }

    result = extract_chapter_timestamp_segments(payload, verse_key="1:1")
    assert result is not None
    assert result.source_type == "qcom_chapter"
    assert result.segment_shape == "3_field"
    assert [segment.word_index for segment in result.segments] == [1, 2]


def test_extract_verse_segments_detects_four_field_shape() -> None:
    payload = {
        "audio_files": [
            {
                "verse_key": "2:255",
                "segments": [
                    [0, 1, 20.0, 80.0],
                    [1, 2, 90.0, 160.0],
                ],
            }
        ]
    }

    result = extract_verse_segments(payload, verse_key="2:255")
    assert result is not None
    assert result.source_type == "qcom_verse"
    assert result.segment_shape == "4_field"
    assert [segment.word_index for segment in result.segments] == [1, 2]
