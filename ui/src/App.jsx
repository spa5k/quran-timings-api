import { useMemo, useRef, useState } from "react";
import "./App.css";

import ActiveAyahPanel from "./components/ActiveAyahPanel.jsx";
import ApiDocsPanel from "./components/ApiDocsPanel.jsx";
import DiagnosticsDrawer from "./components/DiagnosticsDrawer.jsx";
import Masthead from "./components/Masthead.jsx";
import SegmentLedger from "./components/SegmentLedger.jsx";
import TransportCard from "./components/TransportCard.jsx";
import WordInspector from "./components/WordInspector.jsx";
import { usePlaybackController } from "./hooks/usePlaybackController.js";
import { useTimingDeskData } from "./hooks/useTimingDeskData.js";
import { buildDiagnosticsModel } from "./utils/diagnostics.js";
import { formatSurahLabel } from "./utils/surah.js";
import { toneFromRatio } from "./utils/timing.js";

function App() {
  const audioRef = useRef(null);
  const [diagnosticsOpen, setDiagnosticsOpen] = useState(false);

  const {
    recitations,
    reciter,
    setSelectedSurah,
    surahs,
    surahConfig,
    surahMeta,
    timingData,
    resolvedAudioSrc,
    isCatalogLoading,
    isTimingLoading,
    error,
    setError,
    handleReciterChange,
  } = useTimingDeskData();

  const ayahs = useMemo(() => timingData?.ayahs ?? [], [timingData]);
  const words = useMemo(() => timingData?.words ?? [], [timingData]);

  const playback = usePlaybackController({
    audioRef,
    resolvedAudioSrc,
    surahMeta,
    timingData,
    ayahs,
    words,
    setError,
  });

  const diagnostics = useMemo(
    () => buildDiagnosticsModel({ surahMeta, ayahs, words }),
    [surahMeta, ayahs, words],
  );

  if (isCatalogLoading) {
    return (
      <div className="status">
        <div className="status__card">
          <div className="status__k">spa5k/quran-timings-api</div>
          <div className="status__h">Loading Quran Timings Desk…</div>
          <div className="status__sub">Fetching reciters index and surah timing payloads.</div>
        </div>
      </div>
    );
  }

  const currentSurahLabel = surahConfig
    ? formatSurahLabel(surahConfig.number)
    : "No surah selected";

  const hasTimingData = ayahs.length > 0 || words.length > 0;
  const hasAyahAudio = ayahs.some(
    (entry) => typeof entry?.audio_url === "string" && entry.audio_url.trim(),
  );
  const hasSegmentPlayback = Boolean(playback.playerAudioSrc) || hasAyahAudio;

  const qcCoverageTone = toneFromRatio(diagnostics.qcCoverage, 0.98, 0.9);
  const sourceTone = toneFromRatio(diagnostics.sourceCoverage, 0.75, 0.2);
  const warningTone =
    diagnostics.qcWarnings.length === 0
      ? "healthy"
      : diagnostics.qcWarnings.length <= 2
        ? "watch"
        : "risk";
  const sourceP95Tone = Number.isFinite(diagnostics.sourceErrorP95Ms)
    ? diagnostics.sourceErrorP95Ms <= 80
      ? "healthy"
      : diagnostics.sourceErrorP95Ms <= 180
        ? "watch"
        : "risk"
    : "watch";

  return (
    <div className="app">
      <Masthead
        recitations={recitations.map((entry) => ({
          id: entry.slug,
          name: entry.name,
        }))}
        reciterId={reciter?.slug ?? ""}
        onReciterChange={handleReciterChange}
        surahs={surahs.map((entry) => ({ surah: entry.number }))}
        surahValue={surahConfig ? String(surahConfig.number) : ""}
        onSurahChange={(value) => setSelectedSurah(value)}
        currentSurahLabel={currentSurahLabel}
        sourceLabel={diagnostics.sourceLabel}
        isPlaying={playback.isPlaying}
        activeAyah={playback.activeAyah}
        activeWordText={playback.activeWord?.text_uthmani ?? "—"}
        counts={{
          ayahs: ayahs.length,
          words: words.length,
          durationS: playback.durationS,
        }}
      />

      {error ? (
        <div className="toast" role="status">
          <div className="toast__title">API Error</div>
          <div className="toast__body">{error}</div>
        </div>
      ) : null}

      <main className="bench" aria-label="Timing API workbench">
        <SegmentLedger
          ayahs={ayahs}
          activeAyah={playback.activeAyah}
          focusedAyah={playback.focusedAyah}
          hasSourceComparison={diagnostics.hasEveryayahSourceComparison}
          diffByAyah={diagnostics.everyayahDiffByAyah}
          onPlayAyah={playback.playAyah}
        />

        <section className="center" aria-label="Playback desk">
          <TransportCard
            audioSrc={playback.playerAudioSrc}
            audioRef={audioRef}
            isTimingLoading={isTimingLoading}
            isPlaying={playback.isPlaying}
            currentTime={playback.currentTime}
            durationS={playback.durationS}
            isAyahByAyah={playback.isAyahByAyah}
            hasPrevAyah={playback.hasPrevAyah}
            hasNextAyah={playback.hasNextAyah}
            shouldHighlightNextAyah={playback.shouldHighlightNextAyah}
            hasTimingData={hasTimingData}
            hasSegmentPlayback={hasSegmentPlayback}
            onTogglePlayback={playback.togglePlayback}
            onSeek={playback.seekTo}
            onPlayActiveWord={playback.playActiveWord}
            onPlayPrevAyah={playback.playPrevAyah}
            onPlayNextAyah={playback.playNextAyah}
            onStop={playback.stopSegment}
          />

          <ActiveAyahPanel
            ayahNumber={playback.displayAyahNumber}
            words={playback.displayAyahWords}
            activeWordGlobal={playback.activeWordGlobal}
            onWordClick={playback.playWord}
          />
        </section>

        <WordInspector
          words={playback.selectedAyahWords}
          activeWordGlobal={playback.activeWordGlobal}
          focusedAyah={playback.focusedAyah}
          onPlayWord={playback.playWord}
          sourceTimingNote={diagnostics.sourceTimingNote}
        />
      </main>

      <ApiDocsPanel
        reciter={reciter}
        surahConfig={surahConfig}
        surahMeta={surahMeta}
        ayahs={ayahs}
        activeAyah={playback.activeAyah}
      />

      <DiagnosticsDrawer
        open={diagnosticsOpen}
        onToggle={() => setDiagnosticsOpen((value) => !value)}
        hasTimingData={hasTimingData}
        parsedSourceRefs={diagnostics.parsedSourceRefs}
        qc={{
          coverage: diagnostics.qcCoverage,
          coverageTone: qcCoverageTone,
          monotonic: surahMeta?.quality?.monotonic,
          warnings: diagnostics.qcWarnings.length,
          warningTone,
          reasonCodes: diagnostics.qcReasonCodes,
        }}
        source={{
          coverage: diagnostics.sourceCoverage,
          coverageTone: sourceTone,
          matchedWords: diagnostics.sourceComparedWords.length,
          errorMedianMs: diagnostics.sourceErrorMedianMs,
          errorP95Ms: diagnostics.sourceErrorP95Ms,
          p95Tone: sourceP95Tone,
        }}
        everyayah={{
          stitchEval: diagnostics.everyayahStitchEval,
          ranked: diagnostics.everyayahDiffRanked,
          worstBoundaryMs: diagnostics.everyayahWorstBoundaryMs,
          hit80Pct: diagnostics.everyayahBoundaryHit80Pct,
          riskCount: diagnostics.everyayahRiskCount,
        }}
        pipeline={{
          engineName:
            surahMeta?.provenance?.selected_candidate_engine ?? surahMeta?.provenance?.engine?.name,
          segmentSourceType: surahMeta?.provenance?.segment_source_type ?? "none",
          quantizationStepMs: diagnostics.quantizationStepMs,
          speechEndDeltaRatio: diagnostics.speechEndDeltaRatio,
          lexicalMatchRatio: diagnostics.lexicalMatchRatio,
          interpolationRatio: diagnostics.interpolationRatio,
          passTrace: diagnostics.passTrace,
        }}
        distribution={{
          ayahMedianMs: diagnostics.ayahDurationMedianMs,
          ayahP95Ms: diagnostics.ayahDurationP95Ms,
          wordMedianMs: diagnostics.wordDurationMedianMs,
          wordP95Ms: diagnostics.wordDurationP95Ms,
          ayahPauseMedianMs: diagnostics.ayahPauseMedianMs,
        }}
        candidates={{
          entries: diagnostics.candidateScoreEntries,
          topScore: diagnostics.candidateTopScore,
        }}
      />
    </div>
  );
}

export default App;
