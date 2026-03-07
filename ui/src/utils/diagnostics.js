import { clamp01, parseSourceRef, percentile } from "./timing.js";

function deriveSourceLabel(parsedSourceRefs, segmentSourceType) {
  const kinds = new Set(
    parsedSourceRefs
      .map((ref) => ref.kind)
      .filter((kind) => kind === "qcom" || kind === "everyayah"),
  );
  const bits = [];
  if (kinds.has("qcom")) {
    bits.push("Quran.com");
  }
  if (kinds.has("everyayah")) {
    bits.push("EveryAyah");
  }
  if (bits.length) {
    return bits.join(" + ");
  }
  if (segmentSourceType === "qcom_chapter") {
    return "Quran.com (chapter)";
  }
  if (segmentSourceType === "qcom_verse") {
    return "Quran.com (verse)";
  }
  return "None";
}

export function buildDiagnosticsModel({ surahMeta, ayahs, words }) {
  const supervisionSources = Array.isArray(surahMeta?.provenance?.supervision_sources)
    ? surahMeta.provenance.supervision_sources
    : [];
  const parsedSourceRefs = supervisionSources.map((value) => parseSourceRef(value));
  const everyayahSourceRefs = parsedSourceRefs.filter((value) => value.kind === "everyayah");
  const qcomSourceRefs = parsedSourceRefs.filter((value) => value.kind === "qcom");

  const sourceComparedWords = words.filter(
    (word) =>
      Number.isFinite(word?.source_start_s) &&
      Number.isFinite(word?.source_end_s) &&
      !(word.source_start_s === 0 && word.source_end_s === 0) &&
      Number.isFinite(word?.start_s) &&
      Number.isFinite(word?.end_s),
  );

  const sourceBoundaryErrorsMs = [];
  for (const word of sourceComparedWords) {
    sourceBoundaryErrorsMs.push(Math.abs(word.start_s - word.source_start_s) * 1000);
    sourceBoundaryErrorsMs.push(Math.abs(word.end_s - word.source_end_s) * 1000);
  }
  const sourceErrorMedianMs = percentile(sourceBoundaryErrorsMs, 50);
  const sourceErrorP95Ms = percentile(sourceBoundaryErrorsMs, 95);

  const qcWarnings = Array.isArray(surahMeta?.quality?.warnings) ? surahMeta.quality.warnings : [];
  const qcReasonCodes = Array.isArray(surahMeta?.quality?.reason_codes)
    ? surahMeta.quality.reason_codes.filter((value) => String(value).trim())
    : [];

  const everyayahStitchEval =
    surahMeta && typeof surahMeta?.quality?.everyayah_stitch_eval === "object"
      ? surahMeta.quality.everyayah_stitch_eval
      : null;
  const everyayahDiffRows = Array.isArray(everyayahStitchEval?.ayah_differences)
    ? everyayahStitchEval.ayah_differences
    : [];
  const everyayahDiffByAyah = new Map();
  for (const row of everyayahDiffRows) {
    const ayah = Number(row?.ayah);
    if (Number.isFinite(ayah)) {
      everyayahDiffByAyah.set(ayah, row);
    }
  }

  const hasEveryayahSourceComparison =
    everyayahSourceRefs.length > 0 && everyayahDiffByAyah.size > 0;
  const everyayahDiffRanked = everyayahDiffRows
    .map((row) => {
      const absStartMs = Number.isFinite(row.abs_start_ms)
        ? Math.abs(row.abs_start_ms)
        : Math.abs(row.delta_start_ms ?? 0);
      const absEndMs = Number.isFinite(row.abs_end_ms)
        ? Math.abs(row.abs_end_ms)
        : Math.abs(row.delta_end_ms ?? 0);
      const maxAbsBoundaryMs = Math.max(absStartMs, absEndMs);
      return {
        ...row,
        absStartMs,
        absEndMs,
        maxAbsBoundaryMs,
      };
    })
    .sort((first, second) => second.maxAbsBoundaryMs - first.maxAbsBoundaryMs);

  const everyayahWorstBoundaryMs = everyayahDiffRanked.length
    ? everyayahDiffRanked[0].maxAbsBoundaryMs
    : null;
  const everyayahBoundaryHit80Pct = everyayahDiffRanked.length
    ? everyayahDiffRanked.filter((row) => row.maxAbsBoundaryMs <= 80).length /
      everyayahDiffRanked.length
    : null;
  const everyayahRiskCount = everyayahDiffRanked.filter((row) => row.maxAbsBoundaryMs > 250).length;

  const ayahDurationsMs = ayahs
    .map((ayah) => (ayah.end_s - ayah.start_s) * 1000)
    .filter((value) => Number.isFinite(value) && value > 0);
  const wordDurationsMs = words
    .map((word) => (word.end_s - word.start_s) * 1000)
    .filter((value) => Number.isFinite(value) && value > 0);
  const ayahGapDurationsMs = [];
  for (let index = 1; index < ayahs.length; index += 1) {
    const gapMs = (ayahs[index].start_s - ayahs[index - 1].end_s) * 1000;
    if (Number.isFinite(gapMs) && gapMs >= 0) {
      ayahGapDurationsMs.push(gapMs);
    }
  }

  const ayahDurationMedianMs = percentile(ayahDurationsMs, 50);
  const ayahDurationP95Ms = percentile(ayahDurationsMs, 95);
  const wordDurationMedianMs = percentile(wordDurationsMs, 50);
  const wordDurationP95Ms = percentile(wordDurationsMs, 95);
  const ayahPauseMedianMs = percentile(ayahGapDurationsMs, 50);

  const candidateMap =
    surahMeta && typeof surahMeta?.provenance?.candidate_scores === "object"
      ? surahMeta.provenance.candidate_scores
      : null;
  const candidateScoreEntries =
    candidateMap && typeof candidateMap === "object"
      ? Object.entries(candidateMap)
          .filter(([, score]) => Number.isFinite(score))
          .sort((first, second) => second[1] - first[1])
      : [];
  const candidateTopScore = candidateScoreEntries.length ? candidateScoreEntries[0][1] : null;

  const sourceTimingNote = (() => {
    if (sourceComparedWords.length > 0) {
      return null;
    }
    if (everyayahStitchEval) {
      return "Word-level reference is unavailable for EveryAyah; ayah-level stitched comparison is shown below.";
    }
    if (qcomSourceRefs.length > 0) {
      return "Quran.com source found but no word-level source timestamps were attached in this run.";
    }
    if (everyayahSourceRefs.length > 0) {
      return "EveryAyah source detected. Word-level timestamp metadata is not provided by EveryAyah.";
    }
    return "No external source timestamps were attached for this run.";
  })();

  const passTrace = Array.isArray(surahMeta?.provenance?.pass_trace)
    ? surahMeta.provenance.pass_trace
    : [];
  const speechEndDeltaRatio = Number.isFinite(surahMeta?.quality?.speech_end_delta_ratio)
    ? surahMeta.quality.speech_end_delta_ratio
    : null;
  const lexicalMatchRatio = Number.isFinite(surahMeta?.quality?.lexical_match_ratio)
    ? surahMeta.quality.lexical_match_ratio
    : null;
  const quantizationStepMs = Number.isFinite(surahMeta?.quality?.quantization_step_ms)
    ? surahMeta.quality.quantization_step_ms
    : null;
  const interpolationRatio = Number.isFinite(surahMeta?.quality?.interpolated_ratio)
    ? surahMeta.quality.interpolated_ratio
    : null;
  const qcCoverage = Number.isFinite(surahMeta?.quality?.coverage)
    ? clamp01(surahMeta.quality.coverage)
    : null;
  const sourceCoverage = words.length > 0 ? sourceComparedWords.length / words.length : null;
  const sourceLabel = deriveSourceLabel(
    parsedSourceRefs,
    surahMeta?.provenance?.segment_source_type,
  );

  return {
    parsedSourceRefs,
    sourceComparedWords,
    sourceErrorMedianMs,
    sourceErrorP95Ms,
    sourceTimingNote,
    qcomSourceRefs,
    everyayahSourceRefs,
    qcWarnings,
    qcReasonCodes,
    qcCoverage,
    sourceCoverage,
    sourceLabel,
    everyayahStitchEval,
    everyayahDiffByAyah,
    hasEveryayahSourceComparison,
    everyayahDiffRanked,
    everyayahWorstBoundaryMs,
    everyayahBoundaryHit80Pct,
    everyayahRiskCount,
    ayahDurationMedianMs,
    ayahDurationP95Ms,
    wordDurationMedianMs,
    wordDurationP95Ms,
    ayahPauseMedianMs,
    candidateScoreEntries,
    candidateTopScore,
    passTrace,
    speechEndDeltaRatio,
    lexicalMatchRatio,
    quantizationStepMs,
    interpolationRatio,
  };
}
