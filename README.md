# quran-audio-data

Batch Quran recitation timing extraction CLI.

It produces:
- Ayah-level timestamps
- Word-level timestamps
- JSON + CSV outputs

Pipeline policy:
1. Always run strict multi-pass model alignment.
2. Use existing timings as priors only (never final output).
3. Fuse external supervision from EveryAyah/Quran.com when available.

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
- `reference_split` (optional benchmark split tag)

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
uv run qad align --manifest /path/manifest.csv --out /path/out
```

Options:
- `--text-data /path/to/quran_text_uthmani_v1.json`
- `--cache-dir .cache/timings/v3`

`align` always runs the full quality stack (strict multi-engine + multi-pass + supervision fusion).

### Validate outputs

```bash
uv run qad validate --input /path/out
```

### Benchmark sample

```bash
uv run qad benchmark --manifest /path/manifest.csv --out /path/out --sample-size 10
```

### Prepare supervision cache

```bash
uv run qad prepare-supervision --manifest /path/manifest.csv --out-dir .cache/supervision
```

### Validate live supervision endpoints

```bash
uv run qad validate-supervision --reciter-id 2 --chapter 1 --verse-key 1:1
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
- Supports resume mode to reuse existing `reference_templates` and downloaded MP3s
- Writes:
  - `benchmark_manifest.csv`
  - `reference_templates/*.json` (manual labeling templates)
  - `benchmark_metadata.json`

Script equivalent:

```bash
python3 scripts/build_benchmark_data.py --out-dir benchmarks/generated --count 200 --download-audio
```

Optional for stable reciter IDs (instead of auto-sanitized EveryAyah subfolder):

```bash
uv run qad benchmark-data \
  --out-dir benchmarks/generated \
  --reciter-subfolder Abdul_Basit_Murattal_64kbps \
  --manifest-reciter-id abdul_basit_murattal_64kbps \
  --surahs 1
```

### Sync reciter catalog (EveryAyah + Quran.com)

```bash
uv run qad sync-reciters \
  --out data/reciter_catalog.json \
  --enabled-reciters abdul_basit_murattal_64kbps,abdurrahmaan_as-sudays,sa3ood_al-shuraym,yasser_ad-dussary
```

List configured reciters:

```bash
uv run qad list-reciters --enabled-only
```

### Run one reciter + surah end-to-end

```bash
uv run qad run-surah \
  --reciter-id abdurrahmaan_as-sudays \
  --surah 1 \
  --out-root runs/surah_runs
```

This command builds full-surah audio from configured supervision sources only:
- EveryAyah ayah MP3s (concatenated per surah), or
- Quran.com chapter recitation audio API (fallback when EveryAyah is unavailable).

It then runs strict alignment, writes `run_summary.json` with QC/supervision breakdown, and syncs both timing JSON + audio assets into:
- `ui/public/data` (`*_full.json` + `audio/*.mp3`)
- `ui/dist/data` (`*_full.json` + `audio/*.mp3`)

For EveryAyah-backed runs, the pipeline also:
- records per-ayah stitched boundaries in `input/everyayah_stitch_timeline.json`
- evaluates aligned surah-level ayah boundaries against stitched ayah references (raw + offset-normalized metrics in `run_summary.json` under `quality.everyayah_stitch_eval`)

UI now includes a **Data Check** panel that shows:
- output/QC values from generated timing JSON
- supervision source references (`everyayah:*`, `qcom:*`)
- per-word aligned vs source timing deltas when source word timings are available (Quran.com supervision)

### Sync UI data from latest runs

Use this to refresh UI timing JSON files from the newest run artifact for each `reciter_id + surah` pair:

```bash
uv run qad sync-ui-data
```

`sync-ui-data` auto-bootstraps `ui/public/data/catalog.json` from `data/reciter_catalog.json` when needed, and updates `audioSrc` to local `/data/audio/...` files copied from run artifacts.

Useful flags:
- `--dry-run` prints changes without writing files
- `--no-sync-dist` updates `ui/public/data` only
- `--prune-ui` removes stale `*_full.json` and stale `audio/*` files not present in latest run selection

Script equivalent:

```bash
python3 scripts/sync_ui_from_latest_runs.py
```

### Evaluate against references

```bash
uv run qad eval --pred-dir /path/predictions --reference-dir /path/reference --report /path/report.json
```

### Bakeoff evaluation (coverage split + supervision ratios)

```bash
uv run qad eval-bakeoff --pred-dir /path/predictions --reference-dir /path/reference --report /path/bakeoff_report.json
```

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

## Engine lineup

The strict v3 pipeline runs these candidates for fusion:
- `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`
- `nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0`
- WhisperX Arabic aligner

## Output files

Per input row:
- `<stem>.json`
- `<stem>_ayah.csv`
- `<stem>_words.csv`
- `<stem>_qc_report.json`
- `<stem>_text_audit.json`

`<stem>` format:
- `reciter_sNNN_full` for full-surah rows
- `reciter_sNNN_aNNN` for ayah rows

Repository policy:
- Generated artifacts are not stored in git (`runs/`, `benchmarks/generated/`, `ui/public/data/` are rebuildable).
- Regenerate as needed using `qad align`, `qad benchmark-data`, and `qad sync-ui-data`.

## Tests

```bash
uv run --extra dev pytest -q
```

Current suite covers:
- Arabic normalization/token-boundary behavior
- Deterministic canonical word indexing
- External timing validation rejection
- Prior-assisted alignment flow (existing timings as priors)
- Strict candidate fusion + QC rescue behavior
- Supervision segment normalization and endpoint contract parsing
