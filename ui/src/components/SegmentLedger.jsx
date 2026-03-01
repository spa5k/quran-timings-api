import { useMemo, useState } from "react";
import { formatDeltaMs, formatStamp } from "../utils/timing.js";

function asFiniteNumber(value) {
  if (value === null || value === undefined) {
    return null;
  }
  const num = typeof value === "number" ? value : Number(value);
  return Number.isFinite(num) ? num : null;
}

function matchesAyahQuery(ayahNumber, query) {
  if (!query) {
    return true;
  }
  const q = query.trim().toLowerCase();
  if (!q) {
    return true;
  }
  return String(ayahNumber).includes(q) || `ayah ${ayahNumber}`.includes(q);
}

export default function SegmentLedger({
  ayahs,
  activeAyah,
  focusedAyah,
  hasSourceComparison,
  diffByAyah,
  onPlayAyah,
}) {
  const [query, setQuery] = useState("");
  const visible = useMemo(
    () => ayahs.filter((row) => matchesAyahQuery(row.ayah, query)),
    [ayahs, query],
  );

  return (
    <section className="ledger" aria-label="Ayah timeline">
      <header className="ledger__head">
        <div className="ledger__title">
          <h2 className="ledger__h">Ayah Ledger</h2>
          <div className="ledger__meta">
            <span className="tag tag--mono">{ayahs.length} rows</span>
            {hasSourceComparison ? (
              <span className="tag tag--good">EveryAyah diff</span>
            ) : (
              <span className="tag tag--dim">No ref diff</span>
            )}
          </div>
        </div>
        <label className="ledger__search">
          <span className="ledger__searchLabel">Filter</span>
          <input
            className="input"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="e.g. 12"
          />
        </label>
      </header>

      <div className="ledger__scroll" role="list">
        {visible.length ? (
          visible.map((ayah) => {
            const isActive = activeAyah === ayah.ayah;
            const isFocus = focusedAyah === ayah.ayah;

            const sourceAyahRow =
              hasSourceComparison && diffByAyah ? diffByAyah.get(ayah.ayah) : null;
            const sourceStart = asFiniteNumber(sourceAyahRow?.ref_start_s);
            const sourceEnd = asFiniteNumber(sourceAyahRow?.ref_end_s);
            const hasSourceTimes =
              sourceStart !== null &&
              sourceEnd !== null &&
              !(sourceStart === 0 && sourceEnd === 0);
            const sourceDeltaStartMs = asFiniteNumber(sourceAyahRow?.delta_start_ms);
            const sourceDeltaEndMs = asFiniteNumber(sourceAyahRow?.delta_end_ms);

            return (
              <button
                type="button"
                key={`ayah-${ayah.ayah}`}
                role="listitem"
                className={`row ${isActive ? "is-active" : ""} ${
                  isFocus ? "is-focus" : ""
                }`}
                aria-current={isActive ? "true" : undefined}
                onClick={() => onPlayAyah?.(ayah)}
              >
                <div className="row__top">
                  <div className="row__title">Ayah {ayah.ayah}</div>
                  <div className="row__badge" aria-hidden={!isActive}>
                    {isActive ? "LIVE" : ""}
                  </div>
                </div>

                <div className={`row__times ${hasSourceTimes ? "has-ref" : ""}`}>
                  <div className="row__line">
                    <span className="row__k">Our</span>
                    <span className="row__v">
                      {formatStamp(ayah.start_s)} <span aria-hidden="true">→</span>{" "}
                      {formatStamp(ayah.end_s)}
                    </span>
                  </div>

                  {hasSourceTimes ? (
                    <div className="row__line row__line--ref">
                      <span className="row__k">Ref</span>
                      <span className="row__v">
                        {formatStamp(sourceStart)} <span aria-hidden="true">→</span>{" "}
                        {formatStamp(sourceEnd)}
                      </span>
                      <span className="row__delta">
                        Δs {formatDeltaMs(sourceDeltaStartMs)} | Δe{" "}
                        {formatDeltaMs(sourceDeltaEndMs)}
                      </span>
                    </div>
                  ) : null}
                </div>
              </button>
            );
          })
        ) : (
          <div className="empty">No ayahs match this filter.</div>
        )}
      </div>
    </section>
  );
}
