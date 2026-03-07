import { useMemo, useState } from "react";
import { formatStamp } from "../utils/timing.js";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

export default function TransportCard({
  audioSrc,
  audioRef,
  isTimingLoading,
  isPlaying,
  currentTime,
  durationS,
  isAyahByAyah,
  hasPrevAyah,
  hasNextAyah,
  shouldHighlightNextAyah,
  hasTimingData,
  hasSegmentPlayback,
  onTogglePlayback,
  onSeek,
  onPlayActiveWord,
  onPlayPrevAyah,
  onPlayNextAyah,
  onStop,
}) {
  const hasAudio = Boolean(audioSrc);
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
    <section className="transport" aria-label="Playback transport">
      <header className="transport__head">
        <div className="transport__title">
          <div className="transport__kicker">Playback</div>
          <h2 className="transport__h">
            {isTimingLoading ? "Loading timings and audio..." : "Ayah Playback"}
          </h2>
        </div>
        <div className="transport__dial" style={dialStyle} aria-hidden="true">
          <div className="transport__dialInner">
            <div className="transport__dialTime">{formatStamp(safeTime)}</div>
            <div className="transport__dialSub">{isPlaying ? "PLAY" : "PAUSED"}</div>
          </div>
        </div>
      </header>

      <div className="transport__card">
        <audio
          ref={audioRef}
          aria-hidden="true"
          tabIndex={-1}
          preload="auto"
          className="transport__audio transport__audio--hidden"
        />

        <div className="transport__scrub">
          <label className="transport__scrubLabel" htmlFor="scrub">
            <span className="transport__scrubText">{dragging ? "Seek" : "Timeline"}</span>
            <span className="transport__scrubValue">
              {formatStamp(dragging ? draftTime : safeTime)} / {formatStamp(safeDuration)}
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
            disabled={!hasAudio}
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
            onClick={onTogglePlayback}
            disabled={!hasSegmentPlayback}
          >
            {isPlaying ? "Pause" : "Play"}
          </button>
          {isAyahByAyah ? (
            <>
              <button
                type="button"
                className="btn btn--ghost"
                onClick={onPlayPrevAyah}
                disabled={!hasPrevAyah}
              >
                Prev Ayah
              </button>
              <button
                type="button"
                className={`btn btn--ghost ${shouldHighlightNextAyah ? "btn--nudge" : ""}`}
                onClick={onPlayNextAyah}
                disabled={!hasNextAyah}
              >
                Next Ayah
              </button>
            </>
          ) : null}
          <button
            type="button"
            className="btn btn--ghost"
            onClick={onPlayActiveWord}
            disabled={!hasTimingData || !hasSegmentPlayback}
          >
            Play Word
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
        {isAyahByAyah && shouldHighlightNextAyah ? (
          <p className="transport__hint">
            Ayah ended. Tap <strong>Next Ayah</strong> to continue.
          </p>
        ) : null}
      </div>
    </section>
  );
}
