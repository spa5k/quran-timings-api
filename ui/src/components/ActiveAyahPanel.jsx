export default function ActiveAyahPanel({
  ayahNumber,
  words,
  activeWordGlobal,
  onWordClick,
}) {
  return (
    <section className="active" aria-label="Active ayah">
      <header className="active__head">
        <div className="active__label">Now Reciting</div>
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
                title="Play this word"
              >
                {word.text_uthmani}
              </button>
            );
          })}
        </p>
      ) : (
        <p className="active__empty">No ayah text found in this timing set.</p>
      )}
    </section>
  );
}

