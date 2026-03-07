import { formatStamp } from "../utils/timing.js";
import { formatSurahLabel } from "../utils/surah.js";

function Field({ label, id, children }) {
  return (
    <label className="field" htmlFor={id}>
      <span className="field__label">{label}</span>
      {children}
    </label>
  );
}

export default function Masthead({
  recitations,
  reciterId,
  onReciterChange,
  surahs,
  surahValue,
  onSurahChange,
  currentSurahLabel,
  isPlaying,
  activeAyah,
  activeWordText,
  sourceLabel,
  counts,
}) {
  return (
    <header className="mast">
      <div className="mast__brand">
        <div className="mast__seal" aria-hidden="true">
          QTA
        </div>
        <div className="mast__title">
          <p className="mast__kicker">spa5k/quran-timings-api</p>
          <h1 className="mast__h">Quran Timings Desk</h1>
          <p className="mast__sub">Inspect ayah and word timings, drift, and endpoint payloads.</p>
        </div>
      </div>

      <div className="mast__bar" aria-label="Selection controls">
        <Field label="Reciter" id="reciter-select">
          <select
            id="reciter-select"
            className="select"
            value={reciterId ?? ""}
            onChange={(event) => onReciterChange?.(event.target.value)}
          >
            {recitations.map((entry) => (
              <option key={entry.id} value={entry.id}>
                {entry.name}
              </option>
            ))}
          </select>
        </Field>

        <Field label="Surah" id="surah-select">
          <select
            id="surah-select"
            className="select"
            value={surahValue ?? ""}
            onChange={(event) => onSurahChange?.(event.target.value)}
          >
            {surahs.map((entry) => (
              <option key={entry.surah} value={entry.surah}>
                {formatSurahLabel(entry.surah)}
              </option>
            ))}
          </select>
        </Field>

        <div className="mast__stats" aria-label="Run summary">
          <div className="stat">
            <div className="stat__v">{counts?.ayahs ?? 0}</div>
            <div className="stat__k">Ayahs</div>
          </div>
          <div className="stat">
            <div className="stat__v">{counts?.words ?? 0}</div>
            <div className="stat__k">Words</div>
          </div>
          <div className="stat">
            <div className="stat__v">{formatStamp(counts?.durationS ?? 0)}</div>
            <div className="stat__k">Duration</div>
          </div>
        </div>
      </div>

      <div className="mast__chips" aria-live="polite">
        <span className="chip chip--paper">Surah {currentSurahLabel}</span>
        <span className="chip chip--paper">Reference: {sourceLabel ?? "None"}</span>
        <span className={`chip ${isPlaying ? "chip--live" : "chip--idle"}`}>
          {isPlaying ? "Playing" : "Paused"}
        </span>
        <span className="chip chip--ink">Active Ayah: {activeAyah ?? "—"}</span>
        <span className="chip chip--arabic" lang="ar" dir="rtl">
          {activeWordText ?? "—"}
        </span>
      </div>
    </header>
  );
}
