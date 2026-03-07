const REPO_OWNER = "spa5k";
const REPO_NAME = "quran-timings-api";
const DEFAULT_REF = "main";

const PRIMARY_BASE_TEMPLATE = `https://cdn.jsdelivr.net/gh/${REPO_OWNER}/${REPO_NAME}@<version>/data`;
const PRIMARY_BASE = `https://cdn.jsdelivr.net/gh/${REPO_OWNER}/${REPO_NAME}@${DEFAULT_REF}/data`;

const FALLBACK_BASES = [
  `https://raw.githubusercontent.com/${REPO_OWNER}/${REPO_NAME}/${DEFAULT_REF}/data`,
  `https://cdn.statically.io/gh/${REPO_OWNER}/${REPO_NAME}/${DEFAULT_REF}/data`,
  `https://rawcdn.githack.com/${REPO_OWNER}/${REPO_NAME}/${DEFAULT_REF}/data`,
];

function normalizeEndpointPath(path) {
  if (typeof path !== "string" || !path.trim()) {
    return "";
  }
  const value = path.trim();
  if (value.startsWith("http://") || value.startsWith("https://")) {
    try {
      return normalizeEndpointPath(new URL(value).pathname);
    } catch {
      return value;
    }
  }

  let out = value;
  if (out.startsWith("/data/")) {
    out = out.slice("/data".length);
  } else if (out.startsWith("data/")) {
    out = `/${out.slice("data/".length)}`;
  }

  if (!out.startsWith("/")) {
    out = `/${out}`;
  }
  return out;
}

function joinBaseWithEndpoint(base, endpointPath) {
  const endpoint = normalizeEndpointPath(endpointPath);
  return `${String(base).replace(/\/+$/, "")}${endpoint}`;
}

function EndpointRow({ label, path, url }) {
  return (
    <div className="api-docs__row">
      <div className="api-docs__rowLabel">{label}</div>
      <code className="api-docs__code">{path}</code>
      <code className="api-docs__code api-docs__code--url">{url}</code>
    </div>
  );
}

export default function ApiDocsPanel({ reciter, surahConfig, surahMeta, ayahs, activeAyah }) {
  const reciterSlug = reciter?.slug ?? "{slug}";
  const surahNumber = Number.isFinite(surahConfig?.number) ? surahConfig.number : "{surah}";

  const activeAyahNumber = Number.isFinite(Number(activeAyah))
    ? Number(activeAyah)
    : Number.isFinite(Number(ayahs?.[0]?.ayah))
      ? Number(ayahs[0].ayah)
      : null;

  const endpoints = {
    reciters: "/reciters.json",
    reciterMetadata: `/reciters/${reciterSlug}/metadata.json`,
    surahMetadata: normalizeEndpointPath(
      surahConfig?.metadata_endpoint ??
        `/reciters/${reciterSlug}/surahs/${surahNumber}/metadata.json`,
    ),
    surahTimings: normalizeEndpointPath(
      surahConfig?.timings_endpoint ??
        `/reciters/${reciterSlug}/surahs/${surahNumber}/timings.json`,
    ),
  };

  const activeAyahPath =
    activeAyahNumber !== null ? `${endpoints.surahTimings}#ayah=${activeAyahNumber}` : "";
  const activeAyahUrl =
    activeAyahNumber !== null
      ? `${joinBaseWithEndpoint(PRIMARY_BASE, endpoints.surahTimings)}#ayah=${activeAyahNumber}`
      : "";

  const activeAyahRow = Array.isArray(ayahs)
    ? (ayahs.find((row) => Number(row?.ayah) === Number(activeAyahNumber)) ?? null)
    : null;
  const activeAyahAudioUrl =
    typeof activeAyahRow?.audio_url === "string" && activeAyahRow.audio_url.trim()
      ? activeAyahRow.audio_url.trim()
      : "";

  const rows = [
    {
      label: "Reciters index",
      path: endpoints.reciters,
      url: joinBaseWithEndpoint(PRIMARY_BASE, endpoints.reciters),
    },
    {
      label: "Reciter metadata",
      path: endpoints.reciterMetadata,
      url: joinBaseWithEndpoint(PRIMARY_BASE, endpoints.reciterMetadata),
    },
    {
      label: "Surah metadata",
      path: endpoints.surahMetadata,
      url: joinBaseWithEndpoint(PRIMARY_BASE, endpoints.surahMetadata),
    },
    {
      label: "Surah timings",
      path: endpoints.surahTimings,
      url: joinBaseWithEndpoint(PRIMARY_BASE, endpoints.surahTimings),
    },
  ];

  if (activeAyahPath && activeAyahUrl) {
    rows.push({
      label: "Active ayah in timings",
      path: activeAyahPath,
      url: activeAyahUrl,
    });
  }

  if (activeAyahAudioUrl) {
    rows.push({
      label: "Active ayah audio",
      path: `timings.ayahs[ayah=${activeAyahNumber}].audio_url`,
      url: activeAyahAudioUrl,
    });
  }

  const sourceType = surahMeta?.provenance?.segment_source_type;

  return (
    <section className="api-docs" aria-label="API documentation">
      <header className="api-docs__head">
        <div>
          <p className="api-docs__kicker">API</p>
          <h2 className="api-docs__title">Endpoint Quick Reference</h2>
          <p className="api-docs__sub">
            JSON endpoints from <code>spa5k/quran-timings-api</code>.
          </p>
        </div>
        <div className="api-docs__badges">
          <span className="tag tag--paper">{reciterSlug}</span>
          <span className="tag tag--paper">surah {surahNumber}</span>
          {sourceType && sourceType !== "none" ? (
            <span className="tag tag--paper">source: {sourceType}</span>
          ) : null}
        </div>
      </header>

      <div className="api-docs__stack">
        <article className="api-docs__card">
          <h3 className="api-docs__cardTitle">Base URLs</h3>
          <code className="api-docs__code api-docs__block">{PRIMARY_BASE_TEMPLATE}</code>
          <code className="api-docs__code api-docs__block">{PRIMARY_BASE}</code>
          <p className="api-docs__note">
            Set <code>&lt;version&gt;</code> to <code>main</code>, a tag, or a commit.
          </p>
          <details className="api-docs__details">
            <summary>Fallback CDNs</summary>
            <div className="api-docs__rows">
              {FALLBACK_BASES.map((base) => (
                <code key={base} className="api-docs__code api-docs__block">
                  {base}
                </code>
              ))}
            </div>
          </details>
        </article>

        <article className="api-docs__card">
          <h3 className="api-docs__cardTitle">Current Selection Endpoints</h3>
          <div className="api-docs__rows">
            {rows.map((row) => (
              <EndpointRow
                key={`${row.label}-${row.path}`}
                label={row.label}
                path={row.path}
                url={row.url}
              />
            ))}
          </div>
        </article>
      </div>
    </section>
  );
}
