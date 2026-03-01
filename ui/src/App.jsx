import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

import ActiveAyahPanel from "./components/ActiveAyahPanel.jsx";
import DiagnosticsDrawer from "./components/DiagnosticsDrawer.jsx";
import Masthead from "./components/Masthead.jsx";
import SegmentLedger from "./components/SegmentLedger.jsx";
import TransportCard from "./components/TransportCard.jsx";
import WordInspector from "./components/WordInspector.jsx";
import {
  clamp01,
  findSegmentIndexByTime,
  parseSourceRef,
  percentile,
  toneFromRatio,
} from "./utils/timing.js";
import { formatSurahLabel } from "./utils/surah.js";

function App() {
  const audioRef = useRef(null);
  const segmentStopRef = useRef(null);
  const rafRef = useRef(null);
  const lastClockPaintRef = useRef(-1);

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
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);

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
        if (controller.signal.aborted) {
          return;
        }

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
    return () => controller.abort();
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
    return () => controller.abort();
  }, [surahConfig]);

  const ayahs = useMemo(() => timingData?.ayahs ?? [], [timingData]);
  const words = useMemo(() => timingData?.words ?? [], [timingData]);

  const wordsByAyah = useMemo(() => {
    const groups = new Map();
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
        tolerance: 0.03,
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
          !(word.source_start_s === 0 && word.source_end_s === 0) &&
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
  const everyayahDiffByAyah = useMemo(() => {
    const byAyah = new Map();
    for (const row of everyayahDiffRows) {
      const ayah = Number(row?.ayah);
      if (Number.isFinite(ayah)) {
        byAyah.set(ayah, row);
      }
    }
    return byAyah;
  }, [everyayahDiffRows]);
  const hasEveryayahSourceComparison =
    everyayahSourceRefs.length > 0 && everyayahDiffByAyah.size > 0;
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

  const seekTo = useCallback(
    (timeS) => {
      const audio = audioRef.current;
      if (!audio || !Number.isFinite(timeS)) {
        return;
      }
      segmentStopRef.current = null;
      audio.currentTime = Math.max(0, timeS);
      syncFromTime(audio.currentTime);
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
    return (
      <div className="status">
        <div className="status__card">
          <div className="status__k">Quran Audio Data</div>
          <div className="status__h">Loading Timing Desk…</div>
          <div className="status__sub">Fetching catalog and run index.</div>
        </div>
      </div>
    );
  }

  const currentSurahLabel = surahConfig
    ? formatSurahLabel(surahConfig.surah)
    : "No surah selected";

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

  const sourceLabel = (() => {
    const kinds = new Set(
      (parsedSourceRefs ?? [])
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

    const segmentSourceType = timingData?.segment_source_type;
    if (segmentSourceType === "qcom_chapter") {
      return "Quran.com (chapter)";
    }
    if (segmentSourceType === "qcom_verse") {
      return "Quran.com (verse)";
    }
    return "None";
  })();

  return (
    <div className="app">
      <Masthead
        recitations={recitations}
        reciterId={reciter?.id ?? ""}
        onReciterChange={(nextId) => {
          setSelectedReciterId(nextId);
          const nextReciter = recitations.find((entry) => entry.id === nextId);
          const firstSurah = nextReciter?.surahs?.[0];
          setSelectedSurah(firstSurah ? String(firstSurah.surah) : "");
        }}
        surahs={surahs}
        surahValue={surahConfig ? String(surahConfig.surah) : ""}
        onSurahChange={(value) => setSelectedSurah(value)}
        currentSurahLabel={currentSurahLabel}
        sourceLabel={sourceLabel}
        isPlaying={isPlaying}
        activeAyah={activeAyah}
        activeWordText={activeWord?.text_uthmani ?? "—"}
        counts={{
          ayahs: timingData?.ayahs?.length ?? 0,
          words: timingData?.words?.length ?? 0,
          durationS,
        }}
      />

      {error ? (
        <div className="toast" role="status">
          <div className="toast__title">Problem</div>
          <div className="toast__body">{error}</div>
        </div>
      ) : null}

      <main className="bench" aria-label="Timing workbench">
        <SegmentLedger
          ayahs={ayahs}
          activeAyah={activeAyah}
          focusedAyah={focusedAyah}
          hasSourceComparison={hasEveryayahSourceComparison}
          diffByAyah={everyayahDiffByAyah}
          onPlayAyah={playAyah}
        />

        <section className="center" aria-label="Playback desk">
          <TransportCard
            audioKey={surahConfig?.audioSrc ?? "audio"}
            audioSrc={surahConfig?.audioSrc ?? ""}
            audioRef={audioRef}
            isTimingLoading={isTimingLoading}
            isPlaying={isPlaying}
            currentTime={currentTime}
            durationS={durationS}
            hasTimingData={hasTimingData}
            onSeek={seekTo}
            onPlayFullSurah={playFullSurah}
            onPlayActiveAyah={playActiveAyah}
            onPlayActiveWord={playActiveWord}
            onStop={stopSegment}
          />

          <ActiveAyahPanel
            ayahNumber={displayAyahNumber}
            words={displayAyahWords}
            activeWordGlobal={activeWordGlobal}
            onWordClick={playWord}
          />
        </section>

        <WordInspector
          words={selectedAyahWords}
          activeWordGlobal={activeWordGlobal}
          focusedAyah={focusedAyah}
          onPlayWord={playWord}
          sourceTimingNote={sourceTimingNote}
        />
      </main>

      <DiagnosticsDrawer
        open={diagnosticsOpen}
        onToggle={() => setDiagnosticsOpen((value) => !value)}
        hasTimingData={hasTimingData}
        parsedSourceRefs={parsedSourceRefs}
        qc={{
          coverage: qcCoverage,
          coverageTone: qcCoverageTone,
          monotonic: timingData?.qc?.monotonic,
          warnings: qcWarnings.length,
          warningTone,
          reasonCodes: qcReasonCodes,
        }}
        source={{
          coverage: sourceCoverage,
          coverageTone: sourceTone,
          matchedWords: sourceComparedWords.length,
          errorMedianMs: sourceErrorMedianMs,
          errorP95Ms: sourceErrorP95Ms,
          p95Tone: sourceP95Tone,
        }}
        everyayah={{
          stitchEval: everyayahStitchEval,
          ranked: everyayahDiffRanked,
          worstBoundaryMs: everyayahWorstBoundaryMs,
          hit80Pct: everyayahBoundaryHit80Pct,
          riskCount: everyayahRiskCount,
        }}
        pipeline={{
          engineName:
            timingData?.selected_candidate_engine ?? timingData?.engine?.name,
          segmentSourceType: timingData?.segment_source_type ?? "none",
          quantizationStepMs,
          speechEndDeltaRatio,
          lexicalMatchRatio,
          interpolationRatio,
          passTrace,
        }}
        distribution={{
          ayahMedianMs: ayahDurationMedianMs,
          ayahP95Ms: ayahDurationP95Ms,
          wordMedianMs: wordDurationMedianMs,
          wordP95Ms: wordDurationP95Ms,
          ayahPauseMedianMs,
        }}
        candidates={{
          entries: candidateScoreEntries,
          topScore: candidateTopScore,
        }}
      />
    </div>
  );
}

export default App;
