from __future__ import annotations

# Manifest reciter id -> EveryAyah folder defaults.
EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT: dict[str, str] = {
    "abdul_basit_murattal_64kbps": "Abdul_Basit_Murattal_64kbps",
    "abdurrahmaan_as-sudays": "Abdurrahmaan_As-Sudais_192kbps",
    "muhsin_al_qasim": "Muhsin_Al_Qasim_192kbps",
    "sa3ood_al-shuraym": "Saood_ash-Shuraym_128kbps",
    "yasser_ad-dussary": "Yasser_Ad-Dussary_128kbps",
}

# Manifest reciter id -> Quran.com recitation id defaults.
QCOM_RECITATION_ID_BY_RECITER_DEFAULT: dict[str, int] = {
    "abdul_basit_murattal_64kbps": 2,
    "abdurrahmaan_as-sudays": 3,
    "sa3ood_al-shuraym": 10,
}

# Explicit model-only word-supervision branch defaults.
UNSUPPORTED_QCOM_WORD_SUPERVISION_DEFAULT: set[str] = {
    "yasser_ad-dussary",
    "abdulbaset_warsh",
}

# Keep production enablement intentionally narrow by default.
DEFAULT_ENABLED_RECITERS: set[str] = {
    "abdul_basit_murattal_64kbps",
    "abdurrahmaan_as-sudays",
    "muhsin_al_qasim",
    "sa3ood_al-shuraym",
    "yasser_ad-dussary",
}
