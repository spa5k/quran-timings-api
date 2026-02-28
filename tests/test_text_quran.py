from pathlib import Path
import unicodedata

from quran_audio_data.text.quran_text import QuranTextStore, normalize_arabic, tokenize_words


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEXT_DATA = PROJECT_ROOT / "data" / "quran_text_uthmani_v1.json"


def test_normalization_preserves_word_boundaries_for_alignment_tokens() -> None:
    store = QuranTextStore(TEXT_DATA)
    ayah_text = store.get_ayah_text(1, 5)

    original_tokens = tokenize_words(ayah_text)
    normalized_tokens = tokenize_words(normalize_arabic(ayah_text))

    assert len(original_tokens) == len(normalized_tokens)


def test_word_index_mapping_is_deterministic() -> None:
    store = QuranTextStore(TEXT_DATA)
    first = store.build_words(surah=1)
    second = store.build_words(surah=1)

    assert len(first) == len(second)
    assert [w.word_index_global for w in first] == [w.word_index_global for w in second]
    assert [w.ayah for w in first] == [w.ayah for w in second]
    assert [w.word_index_in_ayah for w in first] == [w.word_index_in_ayah for w in second]


def test_normalization_handles_persian_yeh_variants() -> None:
    assert normalize_arabic("یولد") == normalize_arabic("يولد")
    assert normalize_arabic("یكن") == normalize_arabic("يكن")


def test_normalization_removes_quranic_combining_marks() -> None:
    normalized = normalize_arabic("هُدࣰى")
    assert normalized == "هدي"
    assert all(not unicodedata.combining(ch) for ch in normalized)
