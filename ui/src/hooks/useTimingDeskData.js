import { useCallback, useEffect, useMemo, useState } from "react";

function resolveJsonPath(payload, jsonPath) {
  if (!payload || typeof payload !== "object") {
    return null;
  }
  if (typeof jsonPath !== "string" || !jsonPath.startsWith("$.")) {
    return null;
  }
  const parts = jsonPath.slice(2).split(".").filter(Boolean);
  let cursor = payload;
  for (const part of parts) {
    if (!cursor || typeof cursor !== "object" || !(part in cursor)) {
      return null;
    }
    cursor = cursor[part];
  }
  return cursor;
}

function normalizeResolvedAudioUrl(value) {
  if (typeof value !== "string" || !value.trim()) {
    return "";
  }
  const raw = value.trim();
  if (raw.startsWith("http://") || raw.startsWith("https://")) {
    return raw;
  }
  if (raw.startsWith("//")) {
    return `https:${raw}`;
  }
  return `https://verses.quran.com/${raw.replace(/^\/+/, "")}`;
}

function orderedAudioAssetIds(audio) {
  if (!audio || typeof audio !== "object") {
    return [];
  }
  const ids = [];
  if (typeof audio.primary_asset === "string" && audio.primary_asset.trim()) {
    ids.push(audio.primary_asset.trim());
  }
  if (Array.isArray(audio.fallback_order)) {
    for (const id of audio.fallback_order) {
      if (typeof id === "string" && id.trim()) {
        ids.push(id.trim());
      }
    }
  }
  return [...new Set(ids)];
}

export function useTimingDeskData() {
  const [catalog, setCatalog] = useState(null);
  const [selectedReciterId, setSelectedReciterId] = useState("");
  const [selectedSurah, setSelectedSurah] = useState("");
  const [reciterMeta, setReciterMeta] = useState(null);
  const [surahMeta, setSurahMeta] = useState(null);
  const [timingData, setTimingData] = useState(null);
  const [resolvedAudioSrc, setResolvedAudioSrc] = useState("");
  const [isCatalogLoading, setIsCatalogLoading] = useState(true);
  const [isTimingLoading, setIsTimingLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    const controller = new AbortController();

    const loadCatalog = async () => {
      try {
        setIsCatalogLoading(true);
        const response = await fetch("/data/reciters.json", {
          signal: controller.signal,
        });
        if (!response.ok) {
          throw new Error(`reciters index fetch failed (${response.status})`);
        }

        const loaded = await response.json();
        if (controller.signal.aborted) {
          return;
        }

        setCatalog(loaded);
        const firstRecitation = loaded?.reciters?.[0];
        if (firstRecitation) {
          setSelectedReciterId(firstRecitation.slug);
        }
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(`Failed to load reciters index: ${String(loadError)}`);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsCatalogLoading(false);
        }
      }
    };

    loadCatalog();
    return () => controller.abort();
  }, []);

  const recitations = useMemo(() => catalog?.reciters ?? [], [catalog]);
  const reciter = useMemo(
    () => recitations.find((entry) => entry.slug === selectedReciterId) ?? recitations[0] ?? null,
    [recitations, selectedReciterId],
  );

  useEffect(() => {
    const controller = new AbortController();

    const loadReciterMetadata = async () => {
      if (!reciter?.slug) {
        setReciterMeta(null);
        setSelectedSurah("");
        return;
      }

      try {
        const endpoint =
          reciter?.endpoints?.metadata ?? `/data/reciters/${reciter.slug}/metadata.json`;
        const response = await fetch(endpoint, { signal: controller.signal });
        if (!response.ok) {
          throw new Error(`reciter metadata fetch failed (${response.status})`);
        }
        const loaded = await response.json();
        if (controller.signal.aborted) {
          return;
        }
        setReciterMeta(loaded);
        const reciterSurahs = Array.isArray(loaded?.surahs) ? loaded.surahs : [];
        const hasSelected = reciterSurahs.some(
          (entry) => String(entry?.number) === String(selectedSurah),
        );
        if (!hasSelected) {
          const firstSurah = reciterSurahs[0];
          setSelectedSurah(Number.isFinite(firstSurah?.number) ? String(firstSurah.number) : "");
        }
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(`Failed to load reciter metadata: ${String(loadError)}`);
          setReciterMeta(null);
        }
      }
    };

    loadReciterMetadata();
    return () => controller.abort();
  }, [reciter, selectedSurah]);

  const surahs = useMemo(() => reciterMeta?.surahs ?? [], [reciterMeta]);
  const surahConfig = useMemo(
    () =>
      surahs.find((entry) => String(entry?.number) === String(selectedSurah)) ?? surahs[0] ?? null,
    [selectedSurah, surahs],
  );

  useEffect(() => {
    const controller = new AbortController();

    const loadTiming = async () => {
      if (!surahConfig?.timings_endpoint || !surahConfig?.metadata_endpoint) {
        setSurahMeta(null);
        setTimingData(null);
        return;
      }

      try {
        setError("");
        setIsTimingLoading(true);
        const [metadataResponse, timingsResponse] = await Promise.all([
          fetch(surahConfig.metadata_endpoint, { signal: controller.signal }),
          fetch(surahConfig.timings_endpoint, { signal: controller.signal }),
        ]);
        if (!metadataResponse.ok) {
          throw new Error(`surah metadata fetch failed (${metadataResponse.status})`);
        }
        if (!timingsResponse.ok) {
          throw new Error(`surah timings fetch failed (${timingsResponse.status})`);
        }

        const [metadataLoaded, loaded] = await Promise.all([
          metadataResponse.json(),
          timingsResponse.json(),
        ]);
        if (controller.signal.aborted) {
          return;
        }

        setSurahMeta(metadataLoaded);
        setTimingData(loaded);
      } catch (loadError) {
        if (!controller.signal.aborted) {
          setError(`Failed to load timing payload: ${String(loadError)}`);
          setSurahMeta(null);
          setTimingData(null);
        }
      } finally {
        if (!controller.signal.aborted) {
          setIsTimingLoading(false);
        }
      }
    };

    loadTiming();
    return () => controller.abort();
  }, [surahConfig]);

  useEffect(() => {
    const controller = new AbortController();

    const resolveAudioSource = async () => {
      const legacySrc = surahMeta?.audio?.src;
      if (typeof legacySrc === "string" && legacySrc.trim()) {
        setResolvedAudioSrc(legacySrc.trim());
        return;
      }

      const audio = surahMeta?.audio;
      const assets = audio?.assets;
      if (!assets || typeof assets !== "object") {
        setResolvedAudioSrc("");
        return;
      }

      for (const assetId of orderedAudioAssetIds(audio)) {
        const asset = assets?.[assetId];
        if (!asset || typeof asset !== "object") {
          continue;
        }
        const kind = String(asset.kind ?? "");
        if (kind === "direct") {
          const direct = normalizeResolvedAudioUrl(asset.url);
          if (direct) {
            setResolvedAudioSrc(direct);
            return;
          }
          continue;
        }
        if (kind === "resolver") {
          const endpoint = String(asset.url ?? "").trim();
          if (!endpoint) {
            continue;
          }
          try {
            const response = await fetch(endpoint, { signal: controller.signal });
            if (!response.ok) {
              continue;
            }
            const payload = await response.json();
            if (controller.signal.aborted) {
              return;
            }
            const resolved = resolveJsonPath(
              payload,
              String(asset.resolve_json_path ?? "$.audio_file.audio_url"),
            );
            const normalized = normalizeResolvedAudioUrl(resolved);
            if (normalized) {
              setResolvedAudioSrc(normalized);
              return;
            }
          } catch {
            // Skip resolver errors and continue down fallback chain.
          }
        }
      }

      setResolvedAudioSrc("");
    };

    resolveAudioSource();
    return () => controller.abort();
  }, [surahMeta]);

  const handleReciterChange = useCallback((nextId) => {
    setSelectedReciterId(nextId);
    setSelectedSurah("");
    setReciterMeta(null);
    setSurahMeta(null);
    setTimingData(null);
    setResolvedAudioSrc("");
  }, []);

  return {
    recitations,
    reciter,
    selectedSurah,
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
  };
}
