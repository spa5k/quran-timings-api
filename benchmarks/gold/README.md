# Gold Annotation Format

Each JSON file should include word timings for one or more ayahs.

Required keys:
- `meta.reciter_id`
- `meta.surah`
- `words[]` with:
  - `ayah`
  - `word_index_in_ayah`
  - `start_s`
  - `end_s`

Example:

```json
{
  "meta": {
    "reciter_id": "yasser_ad-dussary",
    "surah": 112
  },
  "words": [
    {"ayah": 1, "word_index_in_ayah": 1, "start_s": 0.11, "end_s": 0.18},
    {"ayah": 1, "word_index_in_ayah": 2, "start_s": 0.45, "end_s": 0.62}
  ]
}
```

Use `qad eval --pred-dir <predictions> --gold-dir benchmarks/gold --report <report.json>`
to compute median/p90/p95 boundary error metrics and pass/fail status.
