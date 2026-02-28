import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const TIMING_EPSILON_S = 0.03;

function formatStamp(seconds) {
  const safe = Number.isFinite(seconds) ? Math.max(0, seconds) : 0;
  const mins = Math.floor(safe / 60);
  const secs = Math.floor(safe % 60);
  const millis = Math.floor((safe - Math.floor(safe)) * 1000);
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}.${String(millis).padStart(3, "0")}`;
}

function formatDeltaMs(valueMs) {
  if (!Number.isFinite(valueMs)) {
    return "—";
  }
  const rounded = Math.round(valueMs);
  return `${rounded >= 0 ? "+" : ""}${rounded}ms`;
}

function formatMs(valueMs) {
  if (!Number.isFinite(valueMs)) {
    return "—";
  }
  return `${Math.round(valueMs)}ms`;
}

function formatPercent(value) {
  if (!Number.isFinite(value)) {
    return "—";
  }
  return `${Math.round(value * 100)}%`;
}

function clamp01(value) {
  if (!Number.isFinite(value)) {
    return 0;
  }
  return Math.min(1, Math.max(0, value));
}

function toneFromRatio(value, goodThreshold = 0.9, watchThreshold = 0.7) {
  if (!Number.isFinite(value)) {
    return "watch";
  }
  if (value >= goodThreshold) {
    return "healthy";
  }
  if (value >= watchThreshold) {
    return "watch";
  }
  return "risk";
}

function humanizeSegmentSourceType(value) {
  if (value === "qcom_chapter") {
    return "Quran.com chapter timestamps";
  }
  if (value === "qcom_verse") {
    return "Quran.com verse segments";
  }
  return "No direct segment supervision";
}

function parseSourceRef(value) {
  const raw = String(value ?? "");
  if (raw.startsWith("everyayah:")) {
    const body = raw.slice("everyayah:".length);
    if (body.startsWith("http://") || body.startsWith("https://")) {
      return {
        kind: "everyayah",
        raw,
        label: `EveryAyah file • ${body}`,
      };
    }
    const parts = body.split(":");
    const fields = {};
    for (const part of parts) {
      const idx = part.indexOf("=");
      if (idx <= 0) {
        continue;
      }
      fields[part.slice(0, idx)] = part.slice(idx + 1);
    }
    const bits = [];
    if (fields.subfolder) {
      bits.push(`Folder ${fields.subfolder}`);
    }
    if (fields.surah) {
      bits.push(`Surah ${fields.surah}`);
    }
    if (fields.scope === "full_surah") {
      bits.push("Stitched full-surah reference");
    }
    return {
      kind: "everyayah",
      raw,
      label: bits.join(" • ") || body,
    };
  }

  if (raw.startsWith("qcom:")) {
    const parts = raw.split(":");
    const sourceType = parts[1] ?? "";
    const recitation = parts[2] ?? "";
    const shapeToken = parts.find((item) => item.startsWith("shape="));
    const shape = shapeToken ? shapeToken.slice("shape=".length) : "";
    const sourceLabel =
      sourceType === "qcom_chapter"
        ? "Quran.com chapter timestamps"
        : "Quran.com verse segments";
    const bits = [sourceLabel];
    if (recitation) {
      bits.push(`Recitation ${recitation}`);
    }
    if (shape) {
      bits.push(shape.replace("_", " "));
    }
    return {
      kind: "qcom",
      raw,
      label: bits.join(" • "),
    };
  }

  return {
    kind: "other",
    raw,
    label: raw || "Unknown source reference",
  };
}

function percentile(values, p) {
  if (!Array.isArray(values) || values.length === 0) {
    return null;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const rank = Math.min(
    sorted.length - 1,
    Math.max(0, Math.ceil((p / 100) * sorted.length) - 1),
  );
  return sorted[rank];
}

function findSegmentIndexByTime(
  segments,
  timeS,
  {
    tolerance = TIMING_EPSILON_S,
    holdInGap = false,
    moveToNextInGap = false,
  } = {},
) {
  if (
    !Array.isArray(segments) ||
    segments.length === 0 ||
    !Number.isFinite(timeS)
  ) {
    return -1;
  }

  let left = 0;
  let right = segments.length - 1;
  let candidate = -1;
  const target = timeS + tolerance;

  while (left <= right) {
    const mid = (left + right) >> 1;
    if (segments[mid].start_s <= target) {
      candidate = mid;
      left = mid + 1;
    } else {
      right = mid - 1;
    }
  }

  if (candidate < 0) {
    return -1;
  }

  const current = segments[candidate];
  if (
    timeS >= current.start_s - tolerance &&
    timeS <= current.end_s + tolerance
  ) {
    return candidate;
  }

  const next = segments[candidate + 1];
  if (timeS > current.end_s && (!next || timeS < next.start_s)) {
    if (moveToNextInGap) {
      return next ? candidate + 1 : candidate;
    }
    if (holdInGap) {
      return candidate;
    }
  }

  return -1;
}

// Removed ensureVisible completely to prevent scroll hijacking

function App() {
  const audioRef = useRef(null);
  const segmentStopRef = useRef(null);
  const rafRef = useRef(null);
  const lastClockPaintRef = useRef(-1);
  const ayahListRef = useRef(null);
  const wordListRef = useRef(null);
  const ayahRowRefs = useRef(new Map());
  const wordRowRefs = useRef(new Map());

  const [catalog, setCatalog] = useState(null);
  const [selectedReciterId, setSelectedReciterId] = useState("");
  const [selectedSurah, setSelectedSurah] = useState("");
  const [timingData, setTimingData] = useState(null);

  const [activeAyah, setActiveAyah] = useState(null);
  const [activeWordGlobal, setActiveWordGlobal] = useState(null);
  const [selectedAyah, setSelectedAyah] = useState(null);

  const [currentTime, setCurrentTime] = useState(0);
  const [durationS, setDurationS] = useState(0);
  const [isCatalogLoading, setIsCatalogLoading] = useState(true);
  const [isTimingLoading, setIsTimingLoading] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    const loadCatalog = async () => {
      try {
        setIsCatalogLoading(true);
        const response = await fetch("/data/catalog.json", {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`catalog fetch failed (${response.status})`);
        }

        const loaded = await response.json();
        setCatalog(loaded);

        const firstRecitation = loaded?.recitations?.[0];
        const firstSurah = firstRecitation?.surahs?.[0];

        if (firstRecitation) {
          setSelectedReciterId(firstRecitation.id);
        }
        if (firstSurah) {
          setSelectedSurah(String(firstSurah.surah));
        }
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(`Failed to load catalog: ${String(loadError)}`);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsCatalogLoading(false);
        }
      }
    };

    loadCatalog();
    return () => {
      controller.abort();
    };
  }, []);

  const recitations = useMemo(() => catalog?.recitations ?? [], [catalog]);

  const reciter = useMemo(
    () =>
      recitations.find((entry) => entry.id === selectedReciterId) ??
      recitations[0] ??
      null,
    [recitations, selectedReciterId],
  );

  const surahs = useMemo(() => reciter?.surahs ?? [], [reciter]);

  const surahConfig = useMemo(
    () =>
      surahs.find((entry) => String(entry.surah) === String(selectedSurah)) ??
      surahs[0] ??
      null,
    [selectedSurah, surahs],
  );

  useEffect(() => {
    const controller = new AbortController();

    const loadTiming = async () => {
      if (!surahConfig?.timingJson) {
        setTimingData(null);
        setDurationS(0);
        setSelectedAyah(null);
        setActiveAyah(null);
        setActiveWordGlobal(null);
        return;
      }

      try {
        setError("");
        setIsTimingLoading(true);
        const response = await fetch(surahConfig.timingJson, {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`timing fetch failed (${response.status})`);
        }

        const loaded = await response.json();
        if (controller.signal.aborted) {
          return;
        }

        setTimingData(loaded);
        setDurationS(loaded?.audio?.duration_s ?? 0);

        const firstAyah = loaded?.ayahs?.[0]?.ayah ?? null;
        setSelectedAyah(firstAyah);
        setActiveAyah(firstAyah);
        setActiveWordGlobal(null);
        setCurrentTime(0);
        lastClockPaintRef.current = 0;
        setIsPlaying(false);

        if (audioRef.current) {
          audioRef.current.currentTime = 0;
          audioRef.current.pause();
        }
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(`Failed to load timing data: ${String(loadError)}`);
          setTimingData(null);
          setDurationS(0);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsTimingLoading(false);
        }
      }
    };

    loadTiming();
    return () => {
      controller.abort();
    };
  }, [surahConfig]);

  const ayahs = useMemo(() => timingData?.ayahs ?? [], [timingData]);
  const words = useMemo(() => timingData?.words ?? [], [timingData]);

  const syncFromTime = useCallback(
    (timeS) => {
      if (!Number.isFinite(timeS)) {
        return;
      }

      const audio = audioRef.current;
      if (
        audio &&
        segmentStopRef.current !== null &&
        timeS >= segmentStopRef.current - 0.02
      ) {
        audio.pause();
        segmentStopRef.current = null;
      }

      if (Math.abs(timeS - lastClockPaintRef.current) >= 0.05 || timeS === 0) {
        lastClockPaintRef.current = timeS;
        setCurrentTime(timeS);
      }

      if (!ayahs.length && !words.length) {
        return;
      }

      const ayahIndex = findSegmentIndexByTime(ayahs, timeS, {
        tolerance: TIMING_EPSILON_S,
        holdInGap: true,
      });

      const nextAyah = ayahIndex >= 0 ? ayahs[ayahIndex].ayah : null;
      setActiveAyah((prevAyah) =>
        prevAyah === nextAyah ? prevAyah : nextAyah,
      );

      if (nextAyah !== null) {
        setSelectedAyah((prevAyah) =>
          prevAyah === nextAyah ? prevAyah : nextAyah,
        );
      }

      const wordIndex = findSegmentIndexByTime(words, timeS, {
        tolerance: 0,
        moveToNextInGap: true,
      });
      const nextWord =
        wordIndex >= 0 ? words[wordIndex].word_index_global : null;
      setActiveWordGlobal((prevWord) =>
        prevWord === nextWord ? prevWord : nextWord,
      );
    },
    [ayahs, words],
  );

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) {
      return undefined;
    }

    const stopRaf = () => {
      if (rafRef.current !== null) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };

    const tick = () => {
      syncFromTime(audio.currentTime);
      if (!audio.paused && !audio.ended) {
        rafRef.current = requestAnimationFrame(tick);
      } else {
        rafRef.current = null;
      }
    };

    const startRaf = () => {
      stopRaf();
      rafRef.current = requestAnimationFrame(tick);
    };

    const onPlay = () => {
      setIsPlaying(true);
      startRaf();
    };

    const onPause = () => {
      setIsPlaying(false);
      syncFromTime(audio.currentTime);
      stopRaf();
    };

    const onEnded = () => {
      setIsPlaying(false);
      segmentStopRef.current = null;
      syncFromTime(audio.currentTime);
      stopRaf();
    };

    const onSeek = () => {
      syncFromTime(audio.currentTime);
    };

    const onLoadedMetadata = () => {
      const nextDuration = Number.isFinite(audio.duration)
        ? audio.duration
        : (timingData?.audio?.duration_s ?? 0);
      setDurationS(nextDuration);
      syncFromTime(audio.currentTime);
    };

    const onAudioError = () => {
      setError(
        "Audio failed to load. Please check network access for the audio source.",
      );
    };

    audio.addEventListener("play", onPlay);
    audio.addEventListener("pause", onPause);
    audio.addEventListener("ended", onEnded);
    audio.addEventListener("seeking", onSeek);
    audio.addEventListener("seeked", onSeek);
    audio.addEventListener("timeupdate", onSeek);
    audio.addEventListener("loadedmetadata", onLoadedMetadata);
    audio.addEventListener("error", onAudioError);

    syncFromTime(audio.currentTime);

    return () => {
      audio.removeEventListener("play", onPlay);
      audio.removeEventListener("pause", onPause);
      audio.removeEventListener("ended", onEnded);
      audio.removeEventListener("seeking", onSeek);
      audio.removeEventListener("seeked", onSeek);
      audio.removeEventListener("timeupdate", onSeek);
      audio.removeEventListener("loadedmetadata", onLoadedMetadata);
      audio.removeEventListener("error", onAudioError);
      stopRaf();
    };
  }, [surahConfig?.audioSrc, syncFromTime, timingData?.audio?.duration_s]);

  useEffect(() => {
    if (activeAyah === null) {
      return;
    }
    // const row = ayahRowRefs.current.get(activeAyah);
    // ensureVisible(ayahListRef.current, row, true);
  }, [activeAyah]);

  useEffect(() => {
    if (activeWordGlobal === null) {
      return;
    }
    // const row = wordRowRefs.current.get(activeWordGlobal);
    // ensureVisible(wordListRef.current, row);
  }, [activeWordGlobal, selectedAyah]);

  const wordsByAyah = useMemo(() => {
    const groups = new Map();
    if (!words.length) {
      return groups;
    }

    for (const word of words) {
      if (!groups.has(word.ayah)) {
        groups.set(word.ayah, []);
      }
      groups.get(word.ayah).push(word);
    }

    return groups;
  }, [words]);

  const wordsByGlobal = useMemo(() => {
    const groups = new Map();
    for (const word of words) {
      groups.set(word.word_index_global, word);
    }
    return groups;
  }, [words]);

  const activeWord =
    activeWordGlobal !== null
      ? (wordsByGlobal.get(activeWordGlobal) ?? null)
      : null;
  const focusedAyah = activeWord?.ayah ?? selectedAyah ?? activeAyah;
  const selectedAyahWords = wordsByAyah.get(focusedAyah) ?? [];
  const displayAyahNumber = focusedAyah;
  const displayAyahWords = wordsByAyah.get(displayAyahNumber) ?? [];

  const supervisionSources = useMemo(
    () =>
      Array.isArray(timingData?.supervision_sources)
        ? timingData.supervision_sources
        : [],
    [timingData],
  );

  const parsedSourceRefs = useMemo(
    () => supervisionSources.map((value) => parseSourceRef(value)),
    [supervisionSources],
  );
  const everyayahSourceRefs = useMemo(
    () => parsedSourceRefs.filter((value) => value.kind === "everyayah"),
    [parsedSourceRefs],
  );
  const qcomSourceRefs = useMemo(
    () => parsedSourceRefs.filter((value) => value.kind === "qcom"),
    [parsedSourceRefs],
  );

  const sourceComparedWords = useMemo(
    () =>
      words.filter(
        (word) =>
          Number.isFinite(word?.source_start_s) &&
          Number.isFinite(word?.source_end_s) &&
          Number.isFinite(word?.start_s) &&
          Number.isFinite(word?.end_s),
      ),
    [words],
  );

  const sourceBoundaryErrorsMs = useMemo(() => {
    const out = [];
    for (const word of sourceComparedWords) {
      out.push(Math.abs(word.start_s - word.source_start_s) * 1000);
      out.push(Math.abs(word.end_s - word.source_end_s) * 1000);
    }
    return out;
  }, [sourceComparedWords]);

  const sourceErrorMedianMs = useMemo(
    () => percentile(sourceBoundaryErrorsMs, 50),
    [sourceBoundaryErrorsMs],
  );
  const sourceErrorP95Ms = useMemo(
    () => percentile(sourceBoundaryErrorsMs, 95),
    [sourceBoundaryErrorsMs],
  );
  const sourceBoundaryHit80Pct = useMemo(() => {
    if (!sourceBoundaryErrorsMs.length) {
      return null;
    }
    const hits = sourceBoundaryErrorsMs.filter((value) => value <= 80).length;
    return hits / sourceBoundaryErrorsMs.length;
  }, [sourceBoundaryErrorsMs]);
  const qcWarnings = useMemo(
    () =>
      Array.isArray(timingData?.qc?.warnings) ? timingData.qc.warnings : [],
    [timingData],
  );
  const qcReasonCodes = useMemo(
    () =>
      Array.isArray(timingData?.qc?.reason_codes)
        ? timingData.qc.reason_codes.filter((value) => String(value).trim())
        : [],
    [timingData],
  );
  const everyayahStitchEval = useMemo(
    () =>
      timingData && typeof timingData.everyayah_stitch_eval === "object"
        ? timingData.everyayah_stitch_eval
        : null,
    [timingData],
  );
  const everyayahDiffRows = useMemo(
    () =>
      Array.isArray(everyayahStitchEval?.ayah_differences)
        ? everyayahStitchEval.ayah_differences
        : [],
    [everyayahStitchEval],
  );
  const everyayahDiffRanked = useMemo(() => {
    return everyayahDiffRows
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
      .sort(
        (first, second) => second.maxAbsBoundaryMs - first.maxAbsBoundaryMs,
      );
  }, [everyayahDiffRows]);
  const everyayahWorstBoundaryMs = useMemo(
    () =>
      everyayahDiffRanked.length
        ? everyayahDiffRanked[0].maxAbsBoundaryMs
        : null,
    [everyayahDiffRanked],
  );
  const everyayahBoundaryHit80Pct = useMemo(() => {
    if (!everyayahDiffRanked.length) {
      return null;
    }
    const hits = everyayahDiffRanked.filter(
      (row) => row.maxAbsBoundaryMs <= 80,
    ).length;
    return hits / everyayahDiffRanked.length;
  }, [everyayahDiffRanked]);
  const everyayahRiskCount = useMemo(
    () =>
      everyayahDiffRanked.filter((row) => row.maxAbsBoundaryMs > 250).length,
    [everyayahDiffRanked],
  );
  const ayahDurationsMs = useMemo(
    () =>
      ayahs
        .map((ayah) => (ayah.end_s - ayah.start_s) * 1000)
        .filter((value) => Number.isFinite(value) && value > 0),
    [ayahs],
  );
  const wordDurationsMs = useMemo(
    () =>
      words
        .map((word) => (word.end_s - word.start_s) * 1000)
        .filter((value) => Number.isFinite(value) && value > 0),
    [words],
  );
  const ayahGapDurationsMs = useMemo(() => {
    const out = [];
    for (let index = 1; index < ayahs.length; index += 1) {
      const gapMs = (ayahs[index].start_s - ayahs[index - 1].end_s) * 1000;
      if (Number.isFinite(gapMs) && gapMs >= 0) {
        out.push(gapMs);
      }
    }
    return out;
  }, [ayahs]);
  const ayahDurationMedianMs = useMemo(
    () => percentile(ayahDurationsMs, 50),
    [ayahDurationsMs],
  );
  const ayahDurationP95Ms = useMemo(
    () => percentile(ayahDurationsMs, 95),
    [ayahDurationsMs],
  );
  const wordDurationMedianMs = useMemo(
    () => percentile(wordDurationsMs, 50),
    [wordDurationsMs],
  );
  const wordDurationP95Ms = useMemo(
    () => percentile(wordDurationsMs, 95),
    [wordDurationsMs],
  );
  const ayahPauseMedianMs = useMemo(
    () => percentile(ayahGapDurationsMs, 50),
    [ayahGapDurationsMs],
  );
  const candidateScoreEntries = useMemo(() => {
    const candidateMap =
      timingData && typeof timingData.candidate_scores === "object"
        ? timingData.candidate_scores
        : (timingData?.qc?.engine_candidate_scores ?? null);
    if (!candidateMap || typeof candidateMap !== "object") {
      return [];
    }
    return Object.entries(candidateMap)
      .filter(([, score]) => Number.isFinite(score))
      .sort((first, second) => second[1] - first[1]);
  }, [timingData]);
  const candidateTopScore = useMemo(
    () => (candidateScoreEntries.length ? candidateScoreEntries[0][1] : null),
    [candidateScoreEntries],
  );
  const candidateScoreSpread = useMemo(() => {
    if (candidateScoreEntries.length < 2) {
      return null;
    }
    return (
      candidateScoreEntries[0][1] -
      candidateScoreEntries[candidateScoreEntries.length - 1][1]
    );
  }, [candidateScoreEntries]);
  const sourceTimingNote = useMemo(() => {
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
  }, [
    everyayahSourceRefs.length,
    everyayahStitchEval,
    qcomSourceRefs.length,
    sourceComparedWords.length,
  ]);

  const playSegment = useCallback(
    (startS, endS) => {
      const audio = audioRef.current;
      if (!audio || !Number.isFinite(startS) || !Number.isFinite(endS)) {
        return;
      }

      segmentStopRef.current = endS;
      audio.currentTime = Math.max(0, startS);
      syncFromTime(audio.currentTime);

      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => {});
      }
    },
    [syncFromTime],
  );

  const playFullSurah = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }

    segmentStopRef.current = null;
    audio.currentTime = 0;
    syncFromTime(0);

    const playPromise = audio.play();
    if (playPromise && typeof playPromise.catch === "function") {
      playPromise.catch(() => {});
    }
  }, [syncFromTime]);

  const stopSegment = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }
    audio.pause();
    segmentStopRef.current = null;
    syncFromTime(audio.currentTime);
  }, [syncFromTime]);

  const playAyah = useCallback(
    (ayahEntry) => {
      if (!ayahEntry) {
        return;
      }
      setSelectedAyah(ayahEntry.ayah);
      setActiveAyah(ayahEntry.ayah);
      playSegment(ayahEntry.start_s, ayahEntry.end_s);
    },
    [playSegment],
  );

  const playWord = useCallback(
    (wordEntry) => {
      if (!wordEntry) {
        return;
      }
      setSelectedAyah(wordEntry.ayah);
      setActiveAyah(wordEntry.ayah);
      setActiveWordGlobal(wordEntry.word_index_global);
      playSegment(wordEntry.start_s, wordEntry.end_s);
    },
    [playSegment],
  );

  const playActiveAyah = useCallback(() => {
    if (!ayahs.length) {
      return;
    }

    const match = ayahs.find((ayah) => ayah.ayah === activeAyah) ?? ayahs[0];
    playAyah(match);
  }, [activeAyah, ayahs, playAyah]);

  const playActiveWord = useCallback(() => {
    if (!words.length) {
      return;
    }

    const match =
      words.find((word) => word.word_index_global === activeWordGlobal) ??
      words[0];
    playWord(match);
  }, [activeWordGlobal, playWord, words]);

  if (isCatalogLoading) {
    return <div className="status">Loading timing explorer...</div>;
  }

  const currentSurahLabel = surahConfig
    ? `${surahConfig.surah}. ${surahConfig.title ?? `Surah ${surahConfig.surah}`}`
    : "No surah selected";
  const progressPct =
    durationS > 0 ? Math.min(100, (currentTime / durationS) * 100) : 0;
  const hasTimingData = ayahs.length > 0 || words.length > 0;
  const qcCoverage = Number.isFinite(timingData?.qc?.coverage)
    ? clamp01(timingData.qc.coverage)
    : null;
  const sourceCoverage =
    words.length > 0 ? sourceComparedWords.length / words.length : null;
  const qcCoverageTone = toneFromRatio(qcCoverage, 0.98, 0.9);
  const sourceTone = toneFromRatio(sourceCoverage, 0.75, 0.2);
  const warningTone =
    qcWarnings.length === 0
      ? "healthy"
      : qcWarnings.length <= 2
        ? "watch"
        : "risk";
  const sourceP95Tone = Number.isFinite(sourceErrorP95Ms)
    ? sourceErrorP95Ms <= 80
      ? "healthy"
      : sourceErrorP95Ms <= 180
        ? "watch"
        : "risk"
    : "watch";
  const passTrace = Array.isArray(timingData?.pass_trace)
    ? timingData.pass_trace
    : [];
  const speechEndDeltaRatio = Number.isFinite(
    timingData?.qc?.speech_end_delta_ratio,
  )
    ? timingData.qc.speech_end_delta_ratio
    : null;
  const lexicalMatchRatio = Number.isFinite(timingData?.qc?.lexical_match_ratio)
    ? timingData.qc.lexical_match_ratio
    : null;
  const quantizationStepMs = Number.isFinite(
    timingData?.qc?.quantization_step_ms,
  )
    ? timingData.qc.quantization_step_ms
    : null;
  const interpolationRatio = Number.isFinite(timingData?.qc?.interpolated_ratio)
    ? timingData.qc.interpolated_ratio
    : null;

  return (
    <div className="app-shell">
      <header className="hero">
        <p className="kicker">Quran Audio Data</p>
        <h1>Timing Explorer</h1>
        <p className="subtitle">
          Recitation to Surah to Ayah by Ayah / Word by Word
        </p>
        <div className="hero-meta" aria-live="polite">
          <span className="pill">Surah {currentSurahLabel}</span>
          <span className={`pill ${isPlaying ? "play-state-active" : ""}`}>
            {isPlaying ? "Playing" : "Paused"}
          </span>
          <span className="pill">Active Ayah: {activeAyah ?? "—"}</span>
          <span className="pill arabic-pill">
            {activeWord?.text_uthmani ?? "—"}
          </span>
        </div>
        <section className="active-ayah-panel" aria-label="Active ayah text">
          <p className="active-ayah-panel-label">
            Active Ayah Text{" "}
            {displayAyahNumber ? `(Ayah ${displayAyahNumber})` : ""}
          </p>
          {displayAyahWords.length ? (
            <p className="active-ayah-text" lang="ar" dir="rtl">
              {displayAyahWords.map((word) => (
                <span
                  key={`active-line-word-${word.word_index_global}`}
                  className={`active-ayah-word ${
                    activeWordGlobal === word.word_index_global
                      ? "active-ayah-word-current"
                      : ""
                  }`}
                >
                  {word.text_uthmani}
                </span>
              ))}
            </p>
          ) : (
            <p className="active-ayah-empty">
              No ayah text available for this timing set.
            </p>
          )}
        </section>
      </header>
      <section className="controls">
        <label htmlFor="reciter-select">
          Recitation
          <select
            id="reciter-select"
            value={reciter?.id ?? ""}
            onChange={(event) => {
              const nextId = event.target.value;
              setSelectedReciterId(nextId);
              const nextReciter = recitations.find(
                (entry) => entry.id === nextId,
              );
              const firstSurah = nextReciter?.surahs?.[0];
              setSelectedSurah(firstSurah ? String(firstSurah.surah) : "");
            }}
          >
            {recitations.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
        </label>

        <label htmlFor="surah-select">
          Surah
          <select
            id="surah-select"
            value={surahConfig ? String(surahConfig.surah) : ""}
            onChange={(event) => setSelectedSurah(event.target.value)}
          >
            {surahs.map((entry) => (
              <option key={entry.surah} value={entry.surah}>
                {entry.surah}. {entry.title ?? `Surah ${entry.surah}`}
              </option>
            ))}
          </select>
        </label>

        <div className="stat-grid">
          <div>
            <strong>{timingData?.ayahs?.length ?? 0}</strong>
            <span>Ayahs</span>
          </div>
          <div>
            <strong>{timingData?.words?.length ?? 0}</strong>
            <span>Words</span>
          </div>
          <div>
            <strong>{formatStamp(durationS)}</strong>
            <span>Duration</span>
          </div>
        </div>
      </section>
      {/* --- NEW PLAYER PANEL --- */}
      <section className="bp-panel">
        <div className="bp-head">
          <h2>Playback Interface</h2>
          <p>
            {isTimingLoading
              ? "Syncing timing payload..."
              : `At ${formatStamp(currentTime)}`}
          </p>
        </div>

        <div className="bp-card">
          <audio
            key={surahConfig?.audioSrc ?? "audio"}
            ref={audioRef}
            controls
            preload="metadata"
            src={surahConfig?.audioSrc ?? ""}
            className="bp-audio"
          />

          <div className="bp-timeline-wrap" aria-hidden={!hasTimingData}>
            <div
              className="bp-track"
              role="progressbar"
              aria-label="Playback progress"
              aria-valuemin={0}
              aria-valuemax={100}
              aria-valuenow={Math.round(progressPct)}
            >
              <div
                className="bp-fill"
                style={{ width: `${progressPct}%` }}
              ></div>
            </div>
            <div className="bp-marks">
              <span>{formatStamp(currentTime)}</span>
              <span>{formatStamp(durationS)}</span>
            </div>
          </div>

          <div className="bp-actions">
            <button
              type="button"
              className="bp-btn bp-primary"
              onClick={playFullSurah}
              disabled={!surahConfig?.audioSrc}
            >
              <span className="bp-icon">▶</span> Full Surah
            </button>
            <button
              type="button"
              className="bp-btn bp-secondary"
              onClick={playActiveAyah}
              disabled={!ayahs.length}
            >
              Play Active Ayah
            </button>
            <button
              type="button"
              className="bp-btn bp-secondary"
              onClick={playActiveWord}
              disabled={!words.length}
            >
              Play Active Word
            </button>
            <div className="bp-spacer"></div>
            <button
              type="button"
              className="bp-btn bp-ghost"
              onClick={stopSegment}
              disabled={!hasTimingData}
            >
              ⏏ Stop
            </button>
          </div>
        </div>
      </section>
      {error ? <p className="error">{error}</p> : null}
      {/* --- NEW TIMELINES GRID --- */}
      <main className="bt-grid">
        <section className="bt-panel">
          <div className="bt-head">
            <h2>Ayah Timeline</h2>
            <span className="bt-badge">{ayahs.length} total</span>
          </div>

          <div ref={ayahListRef} className="bt-scroll">
            {ayahs.length ? (
              ayahs.map((ayah) => (
                <button
                  type="button"
                  key={`ayah-${ayah.ayah}`}
                  ref={(node) => {
                    if (node) {
                      ayahRowRefs.current.set(ayah.ayah, node);
                    } else {
                      ayahRowRefs.current.delete(ayah.ayah);
                    }
                  }}
                  className={`bt-row ${activeAyah === ayah.ayah ? "active" : ""} ${
                    focusedAyah === ayah.ayah ? "focused" : ""
                  }`}
                  aria-current={activeAyah === ayah.ayah ? "true" : undefined}
                  onClick={() => playAyah(ayah)}
                >
                  <div className="bt-info">
                    <span className="bt-title">Ayah {ayah.ayah}</span>
                    <span className="bt-time">
                      {formatStamp(ayah.start_s)}{" "}
                      <span className="bt-arrow">→</span>{" "}
                      {formatStamp(ayah.end_s)}
                    </span>
                  </div>
                  {activeAyah === ayah.ayah && (
                    <div className="bt-indicator"></div>
                  )}
                </button>
              ))
            ) : (
              <div className="bt-empty">No ayah timings mapped.</div>
            )}
          </div>
        </section>

        <section className="bt-panel">
          <div className="bt-head">
            <h2>Words {focusedAyah ? `(Ayah ${focusedAyah})` : ""}</h2>
            <span className="bt-badge">{selectedAyahWords.length} subset</span>
          </div>

          <div ref={wordListRef} className="bt-scroll bt-words-scroll">
            {selectedAyahWords.length ? (
              selectedAyahWords.map((word) => (
                <button
                  type="button"
                  key={`word-${word.word_index_global}`}
                  ref={(node) => {
                    if (node) {
                      wordRowRefs.current.set(word.word_index_global, node);
                    } else {
                      wordRowRefs.current.delete(word.word_index_global);
                    }
                  }}
                  className={`bt-row word-row ${activeWordGlobal === word.word_index_global ? "active-word" : ""}`}
                  aria-current={
                    activeWordGlobal === word.word_index_global
                      ? "true"
                      : undefined
                  }
                  onClick={() => playWord(word)}
                >
                  <div className="bt-info">
                    <span className="bt-title arabic">{word.text_uthmani}</span>
                    <div className="bt-meta-col">
                      <span className="bt-time">
                        {formatStamp(word.start_s)}{" "}
                        <span className="bt-arrow">→</span>{" "}
                        {formatStamp(word.end_s)}
                      </span>
                      {Number.isFinite(word?.source_start_s) &&
                      Number.isFinite(word?.source_end_s) ? (
                        <span className="bt-source-time">
                          Src: {formatStamp(word.source_start_s)}{" "}
                          <span className="bt-arrow">→</span>{" "}
                          {formatStamp(word.source_end_s)}
                          <span className="bt-delta">
                            {" "}
                            (Δs{" "}
                            {formatDeltaMs(
                              (word.start_s - word.source_start_s) * 1000,
                            )}{" "}
                            | Δe{" "}
                            {formatDeltaMs(
                              (word.end_s - word.source_end_s) * 1000,
                            )}
                            )
                          </span>
                        </span>
                      ) : null}
                    </div>
                  </div>
                  {activeWordGlobal === word.word_index_global && (
                    <div className="bt-indicator word-indicator"></div>
                  )}
                </button>
              ))
            ) : (
              <div className="bt-empty">
                Select an ayah to view word timing.
              </div>
            )}
          </div>
        </section>
      </main>
      ;
      <section className="bento-wrapper">
        <header className="bento-header">
          <h2>Data Diagnostics</h2>
          <p>
            Quality snapshot for alignment health, source fit, and timing
            behavior.
          </p>
        </header>

        <div className="bento-kpi-row">
          <div className="bento-kpi">
            <span className="bento-kpi-label">QC Cover</span>
            <strong className="bento-kpi-value">
              {formatPercent(qcCoverage)}
            </strong>
            <div
              className={`bento-spark tone-${qcCoverageTone}`}
              style={{
                "--fill": `${Math.round(clamp01(qcCoverage ?? 0) * 100)}%`,
              }}
            ></div>
            <span className="bento-kpi-sub tone-dim">
              {timingData?.qc?.monotonic === false
                ? "Non-monotonic"
                : "Monotonic"}
            </span>
          </div>
          <div className="bento-kpi">
            <span className="bento-kpi-label">Src Match</span>
            <strong className="bento-kpi-value">
              {formatPercent(sourceCoverage)}
            </strong>
            <div
              className={`bento-spark tone-${sourceTone}`}
              style={{
                "--fill": `${Math.round(clamp01(sourceCoverage ?? 0) * 100)}%`,
              }}
            ></div>
            <span className="bento-kpi-sub tone-dim">
              {sourceComparedWords.length} matched
            </span>
          </div>
          <div className="bento-kpi">
            <span className="bento-kpi-label">Alerts</span>
            <strong className="bento-kpi-value">{qcWarnings.length}</strong>
            <div
              className={`bento-spark tone-${warningTone}`}
              style={{
                "--fill": `${qcWarnings.length === 0 ? 100 : Math.min(100, qcWarnings.length * 28)}%`,
              }}
            ></div>
            <span className="bento-kpi-sub tone-dim">
              {qcWarnings.length === 0
                ? "Clean"
                : `${qcReasonCodes.length} codes`}
            </span>
          </div>
          <div className="bento-kpi">
            <span className="bento-kpi-label">Max Drift</span>
            <strong className="bento-kpi-value">
              {formatMs(everyayahWorstBoundaryMs)}
            </strong>
            <div
              className={`bento-spark tone-${sourceP95Tone}`}
              style={{
                "--fill": `${Math.min(100, Number.isFinite(everyayahWorstBoundaryMs) ? (everyayahWorstBoundaryMs / 240) * 100 : 0)}%`,
              }}
            ></div>
            <span className="bento-kpi-sub tone-dim">
              p95 err: {formatMs(sourceErrorP95Ms)}
            </span>
          </div>
        </div>

        {hasTimingData ? (
          <div className="bento-grid">
            <article className="bento-card bento-span-2 bento-card-dark">
              <div className="bc-top">
                <h3>Pipeline Matrix</h3>
                <span
                  className={`bento-chip tone-${timingData?.qc?.monotonic === false ? "risk" : "healthy"}`}
                >
                  {timingData?.qc?.monotonic === false ? "ERR" : "OK"}
                </span>
              </div>
              <div className="bento-matrix">
                <div className="matrix-col">
                  <div className="m-item">
                    <span>Engine</span>{" "}
                    <strong>
                      {timingData?.selected_candidate_engine ??
                        timingData?.engine?.name ??
                        "—"}
                    </strong>
                  </div>
                  <div className="m-item">
                    <span>Supervision</span>{" "}
                    <strong>
                      {humanizeSegmentSourceType(
                        timingData?.segment_source_type ?? "none",
                      )}
                    </strong>
                  </div>
                  <div className="m-item">
                    <span>Quantization</span>{" "}
                    <strong>{formatMs(quantizationStepMs)}</strong>
                  </div>
                </div>
                <div className="matrix-col">
                  <div className="m-item">
                    <span>Speech End Δ</span>{" "}
                    <strong>{formatPercent(speechEndDeltaRatio)}</strong>
                  </div>
                  <div className="m-item">
                    <span>Lexical Match</span>{" "}
                    <strong>{formatPercent(lexicalMatchRatio)}</strong>
                  </div>
                  <div className="m-item">
                    <span>Interpolation</span>{" "}
                    <strong>{formatPercent(interpolationRatio)}</strong>
                  </div>
                </div>
              </div>
              {qcReasonCodes.length > 0 && (
                <div className="bento-tag-array mt-rt">
                  {qcReasonCodes.map((code) => (
                    <span key={code} className="bt-tag">
                      {code}
                    </span>
                  ))}
                </div>
              )}
            </article>

            <article className="bento-card">
              <div className="bc-top">
                <h3>Source Anchors</h3>
              </div>
              <div className="bento-list">
                <div className="bl-row">
                  <span>Audio</span>{" "}
                  <em>{surahConfig?.audioSrc ? "Found" : "Missing"}</em>
                </div>
                <div className="bl-row">
                  <span>Quran.com</span> <em>{qcomSourceRefs.length} refs</em>
                </div>
                <div className="bl-row">
                  <span>EveryAyah</span>{" "}
                  <em>{everyayahSourceRefs.length} refs</em>
                </div>
                <div className="bl-row">
                  <span>Words</span>{" "}
                  <em>
                    {sourceComparedWords.length}/{words.length}
                  </em>
                </div>
                <div className="bl-row">
                  <span>Align Md</span> <em>{formatMs(sourceErrorMedianMs)}</em>
                </div>
              </div>
              {sourceTimingNote && (
                <p className="bento-subtext mt-rt">{sourceTimingNote}</p>
              )}
            </article>

            <article className="bento-card">
              <div className="bc-top">
                <h3>Timing Velocity</h3>
              </div>
              <div className="bento-list mb-rt">
                <div className="bl-row">
                  <span>Refine</span>{" "}
                  <em>{timingData?.qc?.boundary_refine_method ?? "—"}</em>
                </div>
                <div className="bl-row">
                  <span>Med Gap</span> <em>{formatMs(ayahPauseMedianMs)}</em>
                </div>
              </div>
              <div className="bento-track-group">
                <div className="btg-head">
                  <span>Ayah p95</span>{" "}
                  <strong>{formatMs(ayahDurationP95Ms)}</strong>
                </div>
                <div
                  className="btg-bar"
                  style={{
                    "--v-fill": `${Math.min(100, Number.isFinite(ayahDurationP95Ms) ? (ayahDurationP95Ms / 10000) * 100 : 0)}%`,
                  }}
                ></div>
              </div>
              <div className="bento-track-group">
                <div className="btg-head">
                  <span>Word p95</span>{" "}
                  <strong>{formatMs(wordDurationP95Ms)}</strong>
                </div>
                <div
                  className="btg-bar btg-bar-alt"
                  style={{
                    "--v-fill": `${Math.min(100, Number.isFinite(wordDurationP95Ms) ? (wordDurationP95Ms / 1200) * 100 : 0)}%`,
                  }}
                ></div>
              </div>
            </article>

            <article className="bento-card bento-span-full bento-card-dark">
              <div className="bc-top gap-md">
                <h3>Drift Diagnostics</h3>
                {everyayahDiffRanked.length > 0 && (
                  <span
                    className={`bento-chip outline tone-${everyayahRiskCount === 0 ? "healthy" : "watch"}`}
                  >
                    {everyayahRiskCount} risk ayahs
                  </span>
                )}
              </div>

              {everyayahStitchEval ? (
                <div className="bento-drift-layout">
                  <div className="drift-metrics">
                    <div className="dm-block">
                      <span>Covered</span>
                      <strong>
                        {everyayahStitchEval.matched_ayahs ?? "—"} /{" "}
                        {everyayahStitchEval.expected_ayahs ?? "—"}
                      </strong>
                    </div>
                    <div className="dm-block">
                      <span>≤80ms</span>
                      <strong>
                        {formatPercent(everyayahBoundaryHit80Pct)}
                      </strong>
                    </div>
                    <div className="dm-block">
                      <span>Worst</span>
                      <strong className="drift-warn">
                        {formatMs(everyayahWorstBoundaryMs)}
                      </strong>
                    </div>
                    <div className="dm-block">
                      <span>Shift</span>
                      <strong>
                        {formatDeltaMs(
                          Number(everyayahStitchEval.start_offset_s) * 1000,
                        )}
                      </strong>
                    </div>
                  </div>

                  {everyayahDiffRanked.length > 0 && (
                    <div className="drift-leaderboard">
                      {everyayahDiffRanked.slice(0, 5).map((row) => {
                        const rowTone =
                          row.maxAbsBoundaryMs <= 80
                            ? "healthy"
                            : row.maxAbsBoundaryMs <= 250
                              ? "watch"
                              : "risk";
                        return (
                          <div
                            key={`diff-${row.ayah}`}
                            className="drift-lb-item"
                          >
                            <div className="dlb-title">
                              Ayah {row.ayah}{" "}
                              <span className={`tone-${rowTone}`}>
                                • {formatMs(row.maxAbsBoundaryMs)} Δ
                              </span>
                            </div>
                            <div className="dlb-tracks">
                              <div className="dlb-tr">
                                <div
                                  className={`dtb-fill tone-${rowTone}`}
                                  style={{
                                    "--dt-fill": `${Math.min(100, (row.absStartMs / 1200) * 100)}%`,
                                  }}
                                ></div>
                                <span>
                                  {formatDeltaMs(row.delta_start_ms)} <em>S</em>
                                </span>
                              </div>
                              <div className="dlb-tr">
                                <div
                                  className={`dtb-fill tone-${rowTone}`}
                                  style={{
                                    "--dt-fill": `${Math.min(100, (row.absEndMs / 1200) * 100)}%`,
                                  }}
                                ></div>
                                <span>
                                  {formatDeltaMs(row.delta_end_ms)} <em>E</em>
                                </span>
                              </div>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              ) : (
                <p className="bento-subtext">No Eval.</p>
              )}
            </article>

            <article className="bento-card bento-card-darker">
              <div className="bc-top">
                <h3>Candidate Rank</h3>
              </div>
              {candidateScoreEntries.length ? (
                <div className="bento-voters">
                  {candidateScoreEntries.map(([engineName, score]) => (
                    <div key={engineName} className="voter-row">
                      <div className="voter-info">
                        <span>{engineName}</span>{" "}
                        <strong>{score.toFixed(3)}</strong>
                      </div>
                      <div
                        className="voter-bar"
                        style={{
                          "--voter-w": `${Math.min(100, candidateTopScore > 0 ? (score / candidateTopScore) * 100 : 0)}%`,
                        }}
                      ></div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="bento-subtext">None.</p>
              )}
              {passTrace.length > 0 && (
                <div className="bento-tag-array mt-rt">
                  {passTrace.map((step) => (
                    <span key={step} className="bt-tag sm">
                      {step}
                    </span>
                  ))}
                </div>
              )}
            </article>
          </div>
        ) : (
          <div className="bento-placeholder">
            <span className="bp-icon">⛶</span>
            <p>
              Select a reciter with populated timing payload to render the
              diagnostics grid.
            </p>
          </div>
        )}
      </section>
      ;
    </div>
  );
}

export default App;
