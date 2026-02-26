# quran-audio-data

Batch Quran recitation timing extraction CLI.

It produces:
- Ayah-level timestamps
- Word-level timestamps
- JSON + CSV outputs

Pipeline policy:
1. Resolve trusted existing timings first.
2. If missing/invalid, run model alignment.
3. If primary alignment fails or QC fails, run WhisperX fallback.

## Current status

This implementation is production-structured and test-covered, with a practical alignment adapter setup:
- Primary engine wrapper: `NemoAligner` (default: NeMo CTC forced alignment runner)
- Fallback engine: `WhisperXFallbackAligner`
- Existing-source resolver: cache + remote template adapters
- Strict schema + QC gates + CSV exports
- Full-surah quality mode: multi-pass ayah chunk re-alignment with overlap

## Install

Core:

```bash
uv sync
```

Development + tests:

```bash
uv sync --extra dev
```

CPU alignment extras:

```bash
uv sync --extra cpu
```

GPU extras (package deps only):

```bash
uv sync --extra gpu
```

MFA extras:

```bash
uv sync --extra mfa
```

If local `mfa` is not runnable in your Python environment, install Docker and pre-pull the MFA image used by the automatic fallback:

```bash
docker pull mmcauliffe/montreal-forced-aligner:latest
```

For NVIDIA CUDA, install the exact PyTorch wheel for your CUDA runtime from official docs, then re-sync as needed.

## Canonical text data

Canonical text snapshot path:

- `data/quran_text_uthmani_v1.json`

The repository now includes a full corpus (114 surahs / 6,236 ayahs), generated from:

- `https://cdn.jsdelivr.net/gh/fawazahmed0/quran-api@1/editions/ara-quranuthmanienc.json`

To rebuild/update the local canonical file:

```bash
python3 scripts/build_full_corpus.py
```

This writes `data/quran_text_uthmani_v1.json` in the pipeline's canonical schema.

## Manifest format

CSV required columns:
- `audio_path`
- `reciter_id`
- `surah`
- `ayah` (empty means full-surah input)

Optional columns:
- `source_url`
- `sha256`
- `language` (defaults to `ar`)
- `riwaya` (optional text-variant hint)
- `text_variant` (optional canonical variant key)
- `gold_split` (optional benchmark split tag)

Example:

```csv
audio_path,reciter_id,surah,ayah,source_url,sha256,language
/data/audio/001001.mp3,abdulbasit,1,1,,,ar
/data/audio/001_full.mp3,abdulbasit,1,,,,ar
```

## CLI

```bash
uv run qad --help
```

### Align

```bash
uv run qad align --manifest /path/manifest.csv --out /path/out --device auto
```

Options:
- `--engine nemo|whisperx|mfa`
- `--accuracy-mode standard|strict`
- `--device auto|cpu|cuda`
- `--text-data /path/to/quran_text_uthmani_v1.json`
- `--cache-dir .cache/timings`
- `--no-remote`

Default behavior now runs all three engines (`nemo`, `whisperx`, `mfa`) and selects the best result.

### Resolve existing only

```bash
uv run qad resolve-existing --manifest /path/manifest.csv --out /path/out
```

### Validate outputs

```bash
uv run qad validate --input /path/out
```

### Benchmark sample

```bash
uv run qad benchmark --manifest /path/manifest.csv --out /path/out --sample-size 10
```

### Build Benchmark Data (Quran.com + EveryAyah)

```bash
uv run qad benchmark-data \
  --out-dir benchmarks/generated \
  --count 200 \
  --reciter-subfolder Abdul_Basit_Murattal_64kbps \
  --download-audio \
  --timeout-s 30 \
  --request-retries 8 \
  --retry-backoff-s 1 \
  --resume
```

This command:
- Pulls ayah/word metadata from Quran.com API v4 (`/verses/by_key/{surah}:{ayah}`)
- Pulls reciter catalog + ayah MP3s from EveryAyah (`recitations.js` + ayah MP3 URLs)
- Retries transient 429/5xx/timeout failures with exponential backoff
- Supports resume mode to reuse existing `gold_templates` and downloaded MP3s
- Writes:
  - `benchmark_manifest.csv`
  - `gold_templates/*.json` (manual labeling templates)
  - `benchmark_metadata.json`

Script equivalent:

```bash
python3 scripts/build_benchmark_data.py --out-dir benchmarks/generated --count 200 --download-audio
```

### Evaluate against gold

```bash
uv run qad eval --pred-dir /path/predictions --gold-dir /path/gold --report /path/report.json
```

### Validate gold labels

```bash
uv run qad validate-gold --gold-dir /path/gold --report /path/gold_validation_report.json
```

### Auto-label gold (no manual labeling)

```bash
uv run qad auto-label-gold \
  --gold-dir /path/gold_templates \
  --chapter-reciter-id 2 \
  --report /path/auto_label_report.json
```

Notes:
- Uses Quran.com `chapter_recitations/{id}/{surah}?segments=true` word segments.
- For `Abdul_Basit_Murattal_64kbps`, use `--chapter-reciter-id 2`.
- If Quran.com omits a word position in `segments`, boundaries are auto-imputed from neighboring anchors.

### GPU doctor

```bash
uv run qad doctor-gpu
```

## NeMo aligner wiring

`NemoAligner` is preconfigured out-of-the-box to call:

- `python -m quran_audio_data.alignment.nemo_runner`

So no env var is required for the default path.

Default alignment behavior is quality-first:
- Uses NeMo CTC forced alignment (reference-text constrained)
- For full-surah files, runs multi-pass ayah-level refinement on weak segments
- Uses increasing overlap windows per pass before finalizing timestamps

If you want to override the command, set `QAD_NEMO_ALIGN_CMD`:

```bash
export QAD_NEMO_ALIGN_CMD='python /path/to/your_nemo_aligner.py --audio {audio_wav} --transcript {transcript_txt} --model {model} --device {device} --out {output_json}'
```

Template placeholders supported:
- `{audio_wav}`
- `{transcript_txt}`
- `{output_json}`
- `{model}`
- `{device}`

The command must write a JSON containing word timestamps (or `words` list compatible with this project).

If NeMo is unavailable or fails, pipeline can fall back to WhisperX.

## MFA aligner wiring

MFA is enabled by default in the alignment engine set.

Runtime mode:
- If local `mfa` works, it is used directly.
- If local `mfa` is broken/unavailable, Docker fallback is used automatically (`mmcauliffe/montreal-forced-aligner:latest`), so Docker must be installed.
- Default MFA profile uses `english_mfa` with an auto-generated per-file `spn` dictionary (`__auto_spn__`) to keep MFA runnable without extra dictionary setup.
- MFA cache/models are stored under `.cache/mfa`.

Optional one-time model pre-warm:

```bash
docker run --rm -e MFA_ROOT_DIR=/mfa-root -v "$PWD/.cache/mfa:/mfa-root" \
  mmcauliffe/montreal-forced-aligner:latest mfa model download acoustic english_mfa
docker run --rm -e MFA_ROOT_DIR=/mfa-root -v "$PWD/.cache/mfa:/mfa-root" \
  mmcauliffe/montreal-forced-aligner:latest mfa model download dictionary english_us_arpa
```

If your local MFA invocation differs from defaults, set `QAD_MFA_ALIGN_CMD`:

```bash
export QAD_MFA_ALIGN_CMD='mfa align --clean --single_speaker --output_format json {corpus_dir} {dictionary} {acoustic_model} {output_dir}'
```

Template placeholders:
- `{corpus_dir}`
- `{output_dir}`
- `{dictionary}`
- `{acoustic_model}`
- `{audio_wav}`
- `{transcript_txt}`

## Output files

Per input row:
- `<stem>.json`
- `<stem>_ayah.csv`
- `<stem>_words.csv`
- `<stem>_qc_report.json`

`<stem>` format:
- `reciter_sNNN_full` for full-surah rows
- `reciter_sNNN_aNNN` for ayah rows

## Tests

```bash
uv run --extra dev pytest -q
```

Current suite covers:
- Arabic normalization/token-boundary behavior
- Deterministic canonical word indexing
- External timing validation rejection
- Ayah-file existing-resolution flow
- Full-surah existing-resolution flow
- Fallback trigger + schema validation
