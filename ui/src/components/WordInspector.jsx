import { useMemo, useState } from "react";
import { formatDeltaMs, formatStamp } from "../utils/timing.js";

function hasSourceTimes(word) {
  const srcStart = Number(word?.source_start_s);
  const srcEnd = Number(word?.source_end_s);
  if (!Number.isFinite(srcStart) || !Number.isFinite(srcEnd)) {
    return false;
  }
  // Some timing sets fill missing source timestamps with 0/0.
  return !(srcStart === 0 && srcEnd === 0);
}

export default function WordInspector({
  words,
  activeWordGlobal,
  focusedAyah,
  onPlayWord,
  sourceTimingNote,
}) {
  const [onlyWithSource, setOnlyWithSource] = useState(false);
  const visible = useMemo(() => {
    if (!onlyWithSource) {
      return words;
    }
    return words.filter((word) => hasSourceTimes(word));
  }, [onlyWithSource, words]);

  return (
    <section className="inspector" aria-label="Word timeline">
      <header className="inspector__head">
        <div className="inspector__title">
          <h2 className="inspector__h">
            Word Timeline {focusedAyah ? `(Ayah ${focusedAyah})` : ""}
          </h2>
          <div className="inspector__meta">
            <span className="tag tag--mono">{words.length} words</span>
            <label className="toggle">
              <input
                type="checkbox"
                checked={onlyWithSource}
                onChange={(event) => setOnlyWithSource(event.target.checked)}
              />
              <span>Only with source times</span>
            </label>
          </div>
        </div>
        {sourceTimingNote ? <p className="inspector__note">{sourceTimingNote}</p> : null}
      </header>

      <div className="inspector__scroll" role="list">
        {visible.length ? (
          visible.map((word) => {
            const isActive = activeWordGlobal === word.word_index_global;
            const srcStart = Number(word?.source_start_s);
            const srcEnd = Number(word?.source_end_s);
            const hasSrc =
              Number.isFinite(srcStart) &&
              Number.isFinite(srcEnd) &&
              !(srcStart === 0 && srcEnd === 0);

            return (
              <button
                type="button"
                key={`word-${word.word_index_global}`}
                role="listitem"
                className={`wrow ${isActive ? "is-active" : ""}`}
                aria-current={isActive ? "true" : undefined}
                onClick={() => onPlayWord?.(word)}
              >
                <div className="wrow__top">
                  <span className="wrow__text" lang="ar" dir="rtl">
                    {word.text_uthmani}
                  </span>
                  <span className="wrow__badge" aria-hidden={!isActive}>
                    {isActive ? "PLAYING" : ""}
                  </span>
                </div>

                <div className={`wrow__times ${hasSrc ? "has-ref" : ""}`}>
                  <div className={`wrow__line ${hasSrc ? "" : "wrow__line--solo"}`}>
                    {hasSrc ? <span className="wrow__k">API</span> : null}
                    <span className="wrow__v">
                      {formatStamp(word.start_s)} <span aria-hidden="true">→</span>{" "}
                      {formatStamp(word.end_s)}
                    </span>
                  </div>

                  {hasSrc ? (
                    <div className="wrow__line wrow__line--ref">
                      <span className="wrow__k">Ref</span>
                      <span className="wrow__v">
                        {formatStamp(srcStart)} <span aria-hidden="true">→</span>{" "}
                        {formatStamp(srcEnd)}
                      </span>
                      <span className="wrow__delta">
                        Δs {formatDeltaMs((word.start_s - word.source_start_s) * 1000)} | Δe{" "}
                        {formatDeltaMs((word.end_s - word.source_end_s) * 1000)}
                      </span>
                    </div>
                  ) : null}
                </div>
              </button>
            );
          })
        ) : (
          <div className="empty">
            {onlyWithSource
              ? "No source-backed words in this ayah."
              : "Choose an ayah to inspect word timings."}
          </div>
        )}
      </div>
    </section>
  );
}
