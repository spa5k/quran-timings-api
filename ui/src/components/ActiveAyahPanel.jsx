export default function ActiveAyahPanel({ ayahNumber, words, activeWordGlobal, onWordClick }) {
  return (
    <section className="active" aria-label="Active ayah">
      <header className="active__head">
        <div className="active__label">Current Ayah</div>
        <div className="active__meta">
          <span className="tag tag--ink">Ayah {ayahNumber ?? "—"}</span>
          <span className="tag tag--mono">{words?.length ?? 0} words</span>
        </div>
      </header>

      {Array.isArray(words) && words.length ? (
        <p className="active__line" lang="ar" dir="rtl">
          {words.map((word) => {
            const isHot = activeWordGlobal === word.word_index_global;
            return (
              <button
                type="button"
                key={`active-word-${word.word_index_global}`}
                className={`active__word ${isHot ? "is-hot" : ""}`}
                onClick={() => onWordClick?.(word)}
                title="Play word clip"
              >
                {word.text_uthmani}
              </button>
            );
          })}
        </p>
      ) : (
        <p className="active__empty">No ayah text is available for this timing payload.</p>
      )}
    </section>
  );
}
