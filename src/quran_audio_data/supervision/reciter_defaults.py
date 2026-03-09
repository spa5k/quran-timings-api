from __future__ import annotations

# Manifest reciter id -> EveryAyah folder defaults.
EVERYAYAH_SUBFOLDER_BY_RECITER_DEFAULT: dict[str, str] = {
    "eya_abdul_basit_murattal_64kbps": "Abdul_Basit_Murattal_64kbps",
    "eya_abdurrahmaan_as_sudais_192kbps": "Abdurrahmaan_As-Sudais_192kbps",
    "eya_muhsin_al_qasim_192kbps": "Muhsin_Al_Qasim_192kbps",
    "eya_saood_ash_shuraym_128kbps": "Saood_ash-Shuraym_128kbps",
    "eya_yasser_ad_dussary_128kbps": "Yasser_Ad-Dussary_128kbps",
}

# Legacy plain EveryAyah ids that now resolve to source-prefixed catalog ids.
LEGACY_EVERYAYAH_RECITER_ID_ALIASES_DEFAULT: dict[str, str] = {
    "abdul_basit_murattal_64kbps": "eya_abdul_basit_murattal_64kbps",
    "abdurrahmaan_as-sudays": "eya_abdurrahmaan_as_sudais_192kbps",
    "muhsin_al_qasim": "eya_muhsin_al_qasim_192kbps",
    "sa3ood_al-shuraym": "eya_saood_ash_shuraym_128kbps",
    "yasser_ad-dussary": "eya_yasser_ad_dussary_128kbps",
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
    "eya_abdul_basit_murattal_64kbps",
    "eya_abdurrahmaan_as_sudais_192kbps",
    "eya_muhsin_al_qasim_192kbps",
    "eya_saood_ash_shuraym_128kbps",
    "eya_yasser_ad_dussary_128kbps",
}
