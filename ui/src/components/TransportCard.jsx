import { useMemo, useState } from "react";
import { formatStamp } from "../utils/timing.js";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export default function TransportCard({
  audioKey,
  audioSrc,
  audioRef,
  isTimingLoading,
  isPlaying,
  currentTime,
  durationS,
  hasTimingData,
  onSeek,
  onPlayFullSurah,
  onPlayActiveAyah,
  onPlayActiveWord,
  onStop,
}) {
  const safeDuration = Number.isFinite(durationS) && durationS > 0 ? durationS : 0;
  const safeTime = Number.isFinite(currentTime) ? currentTime : 0;
  const progressPct = safeDuration > 0 ? (safeTime / safeDuration) * 100 : 0;

  const [dragging, setDragging] = useState(false);
  const [draftTime, setDraftTime] = useState(safeTime);

  const dialStyle = useMemo(() => {
    const pct = clamp(progressPct, 0, 100);
    return { "--pct": `${pct}%` };
  }, [progressPct]);

  return (
    <section className="transport" aria-label="Playback controls">
      <header className="transport__head">
        <div className="transport__title">
          <div className="transport__kicker">Transport</div>
          <h2 className="transport__h">
            {isTimingLoading ? "Syncing timing payload..." : "Playback Desk"}
          </h2>
        </div>
        <div className="transport__dial" style={dialStyle} aria-hidden="true">
          <div className="transport__dialInner">
            <div className="transport__dialTime">{formatStamp(safeTime)}</div>
            <div className="transport__dialSub">
              {isPlaying ? "LIVE" : "HOLD"}
            </div>
          </div>
        </div>
      </header>

      <div className="transport__card">
        <audio
          key={audioKey}
          ref={audioRef}
          controls
          preload="metadata"
          src={audioSrc ?? ""}
          className="transport__audio"
        />

        <div className="transport__scrub">
          <label className="transport__scrubLabel" htmlFor="scrub">
            <span className="transport__scrubText">
              {dragging ? "Scrub" : "Position"}
            </span>
            <span className="transport__scrubValue">
              {formatStamp(dragging ? draftTime : safeTime)} /{" "}
              {formatStamp(safeDuration)}
            </span>
          </label>
          <input
            id="scrub"
            className="transport__range"
            type="range"
            min={0}
            max={safeDuration || 1}
            step={0.01}
            value={dragging ? draftTime : safeTime}
            disabled={!audioSrc}
            onPointerDown={() => {
              setDragging(true);
              setDraftTime(safeTime);
            }}
            onPointerUp={(event) => {
              const next = Number(event.currentTarget.value);
              setDragging(false);
              setDraftTime(next);
              onSeek?.(next);
            }}
            onKeyUp={(event) => {
              if (event.key === "Enter") {
                onSeek?.(draftTime);
              }
            }}
            onChange={(event) => {
              const next = Number(event.target.value);
              setDraftTime(next);
              if (dragging) {
                return;
              }
              onSeek?.(next);
            }}
          />
          <div className="transport__scrubMarks" aria-hidden="true">
            <span>{formatStamp(0)}</span>
            <span>{formatStamp(safeDuration)}</span>
          </div>
        </div>

        <div className="transport__actions">
          <button
            type="button"
            className="btn btn--primary"
            onClick={onPlayFullSurah}
            disabled={!audioSrc}
          >
            Full Surah
          </button>
          <button
            type="button"
            className="btn btn--ink"
            onClick={onPlayActiveAyah}
            disabled={!hasTimingData}
          >
            Active Ayah
          </button>
          <button
            type="button"
            className="btn btn--ink"
            onClick={onPlayActiveWord}
            disabled={!hasTimingData}
          >
            Active Word
          </button>
          <span className="transport__spacer" />
          <button
            type="button"
            className="btn btn--ghost"
            onClick={onStop}
            disabled={!hasTimingData}
          >
            Stop
          </button>
        </div>
      </div>
    </section>
  );
}
