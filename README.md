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
- Primary engine wrapper: `NemoAligner` (command-template driven via `QAD_NEMO_ALIGN_CMD`)
- Fallback engine: `WhisperXFallbackAligner`
- Existing-source resolver: cache + remote template adapters
- Strict schema + QC gates + CSV exports

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
uv run qad align --manifest /path/manifest.csv --out /path/out --engine nemo --device auto
```

Options:
- `--engine nemo|whisperx`
- `--device auto|cpu|cuda`
- `--text-data /path/to/quran_text_uthmani_v1.json`
- `--cache-dir .cache/timings`
- `--no-remote`

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

## NeMo aligner wiring

`NemoAligner` is preconfigured out-of-the-box to call:

- `python -m quran_audio_data.alignment.nemo_runner`

So no env var is required for the default path.

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
