# Quran Audio Timings API and CLI

In the name of Allah, the Entirely Merciful, the Especially Merciful.

This repository provides a public Quran recitation timings dataset (ayah + word timings) and a CLI to generate/refresh the data.

## Features

- Free static JSON API (CDN-friendly)
- No app-level authentication or rate limits
- Reciter metadata + per-surah timing endpoints
- CLI pipeline to run alignment and publish updated timing files

## UI Screenshots

### Overview (Desktop)

![Quran Audio Timing Desk overview](docs/screenshots/ui-overview-desktop.png)

## Public API

URL structure:

```text
https://cdn.jsdelivr.net/gh/spa5k/quran-timings-api@{version}/data/{endpoint}
```

Versioning:

- `@main` for latest branch
- `@v1`, tags, or commit hash for pinned versions

Base URLs:

- JSDelivr: `https://cdn.jsdelivr.net/gh/spa5k/quran-timings-api@main/data`
- GitHack: `https://rawcdn.githack.com/spa5k/quran-timings-api/main/data`
- Statically: `https://cdn.statically.io/gh/spa5k/quran-timings-api/main/data`
- GitHub Raw: `https://raw.githubusercontent.com/spa5k/quran-timings-api/main/data`
- GitLoaf: `https://gitloaf.com/cdn/spa5k/quran-timings-api/main/data`

### Endpoints

1. `/reciters.json`  
   List reciters, source pools, and counts.

   Example:  
   `https://cdn.jsdelivr.net/gh/spa5k/quran-timings-api@main/data/reciters.json`

2. `/api/reciters/{slug}/metadata.json`  
   Reciter-level metadata + available surahs.

   Example:  
   `https://cdn.jsdelivr.net/gh/spa5k/quran-timings-api@main/data/api/reciters/yasser_ad-dussary/metadata.json`

3. `/api/reciters/{slug}/surahs/{surah}/metadata.json`  
   Surah-level metadata (counts, duration, audio contract).

   Example:  
   `https://cdn.jsdelivr.net/gh/spa5k/quran-timings-api@main/data/api/reciters/yasser_ad-dussary/surahs/114/metadata.json`

4. `/api/reciters/{slug}/surahs/{surah}/timings.json`  
   Ayah and word timing payload.

   Example:  
   `https://cdn.jsdelivr.net/gh/spa5k/quran-timings-api@main/data/api/reciters/yasser_ad-dussary/surahs/114/timings.json`

### Quick API usage

```bash
curl -s https://cdn.jsdelivr.net/gh/spa5k/quran-timings-api@main/data/reciters.json | jq '.counts'
```

```bash
curl -s https://cdn.jsdelivr.net/gh/spa5k/quran-timings-api@main/data/api/reciters/muhsin_al_qasim/surahs/113/timings.json | jq '.ayahs[0]'
```

## Available Reciters

Snapshot below is from `data/reciters.json` generated on `2026-03-06`.

### Enabled reciters in the current API dataset

| Slug                   | Name                 | Check Type     | EveryAyah Subfolder         | Quran.com Recitation ID | Surahs Available            |
| ---------------------- | -------------------- | -------------- | --------------------------- | ----------------------- | --------------------------- |
| `eya_nasser_alqatami`  | eya_nasser_alqatami  | `model_only`   | -                           | -                       | 90, 110, 111, 112, 113, 114 |
| `eya_salah_al_budair`  | eya_salah_al_budair  | `model_only`   | -                           | -                       | 90, 110, 111, 112, 113, 114 |
| `muhsin_al_qasim`      | Muhsin Al Qasim      | `ayah_by_ayah` | `Muhsin_Al_Qasim_192kbps`   | -                       | 110, 111, 112, 113, 114     |
| `qcom_husary`          | qcom_husary          | `model_only`   | -                           | -                       | 90, 110, 111, 112, 113, 114 |
| `qcom_mishari_alafasy` | qcom_mishari_alafasy | `model_only`   | -                           | -                       | 90, 110, 111, 112, 113, 114 |
| `yasser_ad-dussary`    | Yasser_Ad-Dussary    | `ayah_by_ayah` | `Yasser_Ad-Dussary_128kbps` | -                       | 110, 111, 112, 113, 114     |

### Source coverage and rollout

- `data/reciters.json` is filled with reciters sourced from:
  - Quran.com
  - EveryAyah
  - Quranicaudio.com
- Source lists are published in the reciter index (for example: `everyayah_reciters`, `quran_com_reciters`), and timing coverage is being added gradually.
- Surah timings are published incrementally per reciter (surah-by-surah), so `surahs_available` grows over time as new timings are contributed.

## Run the CLI Locally

### Prerequisites

- Python `3.12+`
- [`uv`](https://docs.astral.sh/uv/)

### Install and run

```bash
git clone https://github.com/spa5k/quran-timings-api.git
cd quran-timings-api
uv sync
uv run qad --help
```

Optional extras:

```bash
uv sync --extra dev   # tests/lint
uv sync --extra cpu   # CPU alignment deps
uv sync --extra gpu   # GPU alignment deps
uv sync --extra mfa   # Montreal Forced Aligner
```

If you want `qad` directly in your active environment:

```bash
uv venv
source .venv/bin/activate
uv pip install -e .
qad --help
```

### Main commands

```bash
uv run qad sync-reciters
uv run qad list-reciters --enabled-only
uv run qad run-surah --reciter-id yasser_ad-dussary --surah 114
uv run qad build-api
```

Non-interactive API build example:

```bash
uv run qad build-api \
  --no-interactive \
  --reciters yasser_ad-dussary,muhsin_al_qasim \
  --surahs 110-114 \
  --runs-root runs \
  --out-root runs/api_build \
  --api-root data/api \
  --ui-data-dir ui/public/data \
  --force
```

Export-only (reuse existing run artifacts):

```bash
uv run qad build-api --export-only --no-interactive
```

## Contribute Your Timings

Contributions for new or improved timings are welcome.

1. Refresh reciter catalog:

```bash
uv run qad sync-reciters
```

2. Run timing generation for target reciter/surah:

```bash
uv run qad run-surah --reciter-id <reciter_slug> --surah <1-114>
```

3. Export API artifacts:

```bash
uv run qad build-api --export-only --no-interactive
```

4. Open a PR with updated files under:

- `data/api/reciters/...`
- `data/reciters.json` (if reciter metadata changed)

5. Include what changed in the PR description:

- Reciter slug(s)
- Surah number(s)
- Any source/audio notes

## Development / Tests

```bash
uv run --extra dev pytest -q
```

## Share

Please share the project and star the repository if it helps you.
