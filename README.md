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

Only one public command is exposed: `qad detect`.

### Run mode (default)

```bash
uv run qad detect \
  --audio-url https://download.quranicaudio.com/quran/abdullaah_3awwaad_al-juhaynee/001.mp3 \
  --reciter-id quranicaudio_abdullaah_3awwaad_al_juhaynee \
  --surah 1
```

Required:
- `--audio-url <url>`
- `--reciter-id <id>`
- `--surah <1..114>`

Optional:
- `--ayah <n>` (if omitted, input is treated as full-surah audio)
- `--out-root runs/detect`
- `--cache-dir .cache/timings/v3`
- `--text-data data/quran_text_uthmani_v1.json`

No automatic surah/ayah detection is performed. You must pass `--surah` and optionally `--ayah`.

Reciter checks in run mode:
- Source of truth: `data/reciters.json` only
- Prefilled from:
  - Quranicaudio (`quranicaudio.com`)
  - EveryAyah (`everyayah.com`)
  - Quran.com (`api.quran.com`)
 - Prefilled IDs are source-scoped slugs, for example:
  - `quranicaudio_abdullaah_3awwaad_al_juhaynee`
  - `everyayah_abdurrahmaan_as_sudais_192kbps`
  - `quran_com_7`
- Unknown reciter:
  - Interactive shell: prompts inline to create it, then continues.
  - Non-interactive shell: exits with a setup command example.

### Setup mode (create/update reciter)

```bash
uv run qad detect \
  --setup-reciter \
  --reciter-id quranicaudio_abdullaah_3awwaad_al_juhaynee \
  --reciter-name "Abdullaah 3awwaad Al-Juhaynee" \
  --source quranicaudio \
  --notes "Main download.quranicaudio.com reciter slug"
```

Optional fields (`--reciter-name`, `--source`, `--notes`) are prompted in interactive mode when omitted.

### List mode

```bash
uv run qad detect --list-reciters
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
- Regenerate as needed using `qad detect`.

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
