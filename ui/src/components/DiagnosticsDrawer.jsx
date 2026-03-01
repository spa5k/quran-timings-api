import {
  formatDeltaMs,
  formatMs,
  formatPercent,
  humanizeSegmentSourceType,
} from "../utils/timing.js";

function ToneDot({ tone }) {
  return <span className={`tone tone--${tone ?? "watch"}`} aria-hidden="true" />;
}

function Kpi({ label, value, sub, tone }) {
  return (
    <div className="kpi">
      <div className="kpi__top">
        <span className="kpi__k">{label}</span>
        <ToneDot tone={tone} />
      </div>
      <div className="kpi__v">{value}</div>
      <div className="kpi__sub">{sub}</div>
    </div>
  );
}

export default function DiagnosticsDrawer({
  open,
  onToggle,
  hasTimingData,
  parsedSourceRefs,
  qc,
  source,
  everyayah,
  pipeline,
  distribution,
  candidates,
}) {
  return (
    <section className={`drawer ${open ? "is-open" : ""}`}>
      <button type="button" className="drawer__toggle" onClick={onToggle}>
        <span className="drawer__toggleTitle">Diagnostics</span>
        <span className="drawer__toggleMeta">
          {open ? "Close" : "Open"} panel
        </span>
      </button>

      {open ? (
        <div className="drawer__body">
          <header className="drawer__head">
            <div>
              <h2 className="drawer__h">Data Diagnostics</h2>
              <p className="drawer__sub">
                QC coverage, source fit, drift, and timing distribution.
              </p>
            </div>
            <div className="drawer__sources" aria-label="Supervision sources">
              {(parsedSourceRefs ?? []).length ? (
                parsedSourceRefs.slice(0, 4).map((ref) => (
                  <span key={ref.raw} className="chip chip--paper">
                    {ref.label}
                  </span>
                ))
              ) : (
                <span className="chip chip--idle">No sources attached</span>
              )}
            </div>
          </header>

          <div className="kpiRow">
            <Kpi
              label="QC Cover"
              value={formatPercent(qc?.coverage)}
              sub={qc?.monotonic === false ? "Non-monotonic" : "Monotonic"}
              tone={qc?.coverageTone}
            />
            <Kpi
              label="Src Match"
              value={formatPercent(source?.coverage)}
              sub={`${source?.matchedWords ?? 0} matched`}
              tone={source?.coverageTone}
            />
            <Kpi
              label="Alerts"
              value={String(qc?.warnings ?? 0)}
              sub={(qc?.reasonCodes ?? []).length ? `${(qc?.reasonCodes ?? []).length} codes` : "Clean"}
              tone={qc?.warningTone}
            />
            <Kpi
              label="Max Drift"
              value={formatMs(everyayah?.worstBoundaryMs)}
              sub={`p95 err: ${formatMs(source?.errorP95Ms)}`}
              tone={source?.p95Tone}
            />
          </div>

          {hasTimingData ? (
            <div className="diagGrid">
              <article className="card card--ink card--span2">
                <header className="card__head">
                  <h3 className="card__h">Pipeline Matrix</h3>
                  <span className="tag tag--mono">
                    {qc?.monotonic === false ? "ERR" : "OK"}
                  </span>
                </header>
                <div className="matrix">
                  <div className="matrix__col">
                    <div className="kv">
                      <span>Engine</span>
                      <strong>{pipeline?.engineName ?? "—"}</strong>
                    </div>
                    <div className="kv">
                      <span>Supervision</span>
                      <strong>
                        {humanizeSegmentSourceType(pipeline?.segmentSourceType ?? "none")}
                      </strong>
                    </div>
                    <div className="kv">
                      <span>Quantization</span>
                      <strong>{formatMs(pipeline?.quantizationStepMs)}</strong>
                    </div>
                  </div>
                  <div className="matrix__col">
                    <div className="kv">
                      <span>Speech End Δ</span>
                      <strong>{formatPercent(pipeline?.speechEndDeltaRatio)}</strong>
                    </div>
                    <div className="kv">
                      <span>Lexical Match</span>
                      <strong>{formatPercent(pipeline?.lexicalMatchRatio)}</strong>
                    </div>
                    <div className="kv">
                      <span>Interpolation</span>
                      <strong>{formatPercent(pipeline?.interpolationRatio)}</strong>
                    </div>
                  </div>
                </div>
                {(qc?.reasonCodes ?? []).length ? (
                  <div className="tags">
                    {qc.reasonCodes.map((code) => (
                      <span key={code} className="tag tag--paper">
                        {code}
                      </span>
                    ))}
                  </div>
                ) : null}
              </article>

              <article className="card card--paper">
                <header className="card__head">
                  <h3 className="card__h">Timing Distribution</h3>
                </header>
                <div className="stack">
                  <div className="kv">
                    <span>Ayah median</span>
                    <strong>{formatMs(distribution?.ayahMedianMs)}</strong>
                  </div>
                  <div className="kv">
                    <span>Ayah p95</span>
                    <strong>{formatMs(distribution?.ayahP95Ms)}</strong>
                  </div>
                  <div className="kv">
                    <span>Word median</span>
                    <strong>{formatMs(distribution?.wordMedianMs)}</strong>
                  </div>
                  <div className="kv">
                    <span>Word p95</span>
                    <strong>{formatMs(distribution?.wordP95Ms)}</strong>
                  </div>
                  <div className="kv">
                    <span>Ayah pause median</span>
                    <strong>{formatMs(distribution?.ayahPauseMedianMs)}</strong>
                  </div>
                </div>
              </article>

              <article className="card card--paper card--span2">
                <header className="card__head">
                  <h3 className="card__h">Drift Diagnostics</h3>
                  {Number.isFinite(everyayah?.riskCount) ? (
                    <span className="tag tag--mono">{everyayah.riskCount} risk ayahs</span>
                  ) : null}
                </header>

                {everyayah?.stitchEval ? (
                  <div className="drift">
                    <div className="drift__stats">
                      <div className="kv">
                        <span>Covered</span>
                        <strong>
                          {everyayah.stitchEval.matched_ayahs ?? "—"} /{" "}
                          {everyayah.stitchEval.expected_ayahs ?? "—"}
                        </strong>
                      </div>
                      <div className="kv">
                        <span>≤80ms</span>
                        <strong>{formatPercent(everyayah?.hit80Pct)}</strong>
                      </div>
                      <div className="kv">
                        <span>Worst</span>
                        <strong className="warn">{formatMs(everyayah?.worstBoundaryMs)}</strong>
                      </div>
                      <div className="kv">
                        <span>Shift</span>
                        <strong>{formatDeltaMs(Number(everyayah.stitchEval.start_offset_s) * 1000)}</strong>
                      </div>
                    </div>

                    {(everyayah?.ranked ?? []).length ? (
                      <div className="drift__board">
                        {everyayah.ranked.slice(0, 6).map((row) => {
                          const rowTone =
                            row.maxAbsBoundaryMs <= 80
                              ? "healthy"
                              : row.maxAbsBoundaryMs <= 250
                                ? "watch"
                                : "risk";
                          return (
                            <div key={`diff-${row.ayah}`} className="driftRow">
                              <div className="driftRow__title">
                                Ayah {row.ayah} <span className={`toneText toneText--${rowTone}`}>{formatMs(row.maxAbsBoundaryMs)}</span>
                              </div>
                              <div className="driftRow__bars">
                                <div className={`bar bar--${rowTone}`} style={{ "--w": `${Math.min(100, (row.absStartMs / 1200) * 100)}%` }}>
                                  <span>{formatDeltaMs(row.delta_start_ms)} <em>S</em></span>
                                </div>
                                <div className={`bar bar--${rowTone}`} style={{ "--w": `${Math.min(100, (row.absEndMs / 1200) * 100)}%` }}>
                                  <span>{formatDeltaMs(row.delta_end_ms)} <em>E</em></span>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    ) : (
                      <p className="muted">No drift rows.</p>
                    )}
                  </div>
                ) : (
                  <p className="muted">No EveryAyah stitch evaluation present.</p>
                )}
              </article>

              <article className="card card--ink">
                <header className="card__head">
                  <h3 className="card__h">Candidate Rank</h3>
                </header>
                {(candidates?.entries ?? []).length ? (
                  <div className="cand">
                    {candidates.entries.map(([engineName, score]) => (
                      <div key={engineName} className="candRow">
                        <div className="candRow__top">
                          <span>{engineName}</span>
                          <strong>{Number(score).toFixed(3)}</strong>
                        </div>
                        <div
                          className="candRow__bar"
                          style={{
                            "--w": `${Math.min(100, candidates.topScore > 0 ? (score / candidates.topScore) * 100 : 0)}%`,
                          }}
                        />
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="muted">No candidate score map found.</p>
                )}

                {(pipeline?.passTrace ?? []).length ? (
                  <div className="tags">
                    {pipeline.passTrace.map((step) => (
                      <span key={step} className="tag tag--paper">
                        {step}
                      </span>
                    ))}
                  </div>
                ) : null}
              </article>
            </div>
          ) : (
            <div className="placeholder">
              Select a reciter/surah with a populated timing payload to render diagnostics.
            </div>
          )}
        </div>
      ) : null}
    </section>
  );
}
